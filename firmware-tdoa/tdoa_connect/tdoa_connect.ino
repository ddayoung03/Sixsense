#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "driver/i2s.h"
#include "driver/rtc_io.h"
#include "arduinoFFT.h"

// ===== 딥슬립 핀 정의 =====
// BUTTON_PIN (GPIO12 / A5): 택트 스위치. LOW로 깨어남(ext0).
// LED_PIN    (GPIO48)     : 긱블 나노 S3 빌트인 LED. 켜짐=활성, 꺼짐=수면.
#define BUTTON_PIN 12
#define LED_PIN    48

// ===== 하드웨어 핀 정의 (I2S, 긱블 나노 기준) =====
#define I2S0_WS   D2
#define I2S0_SCK  D3
#define I2S0_SD   D4

#define I2S1_WS   D6
#define I2S1_SCK  D7
#define I2S1_SD   D5

// ===== UART 통신 핀 정의 (보드 간 연결, 긱블 나노 맞춤형) =====
#define UART_TX   TX
#define UART_RX   RX

// ===== 오디오 및 TDOA 상수 =====
#define FS                  16000.0f
#define SAMPLES_PER_READ    1024
#define DISTANCE_MICS_M     0.25f
#define SOUND_SPEED_MPS     343.0f
// 💡 만약 너무 작은 소리에도 반응해서 각도가 튄다면 이 값을 키우고(예: 1.0e-5f),
// 소리를 잘 못 잡으면 이 값을 줄여주세요(예: 1.0e-6f).
#define ENERGY_THRESH       5.0e-6f
// 메인 상관 피크가 (PSR_GUARD 샘플 이상 떨어진) 최대 부엽보다 이 배수 이상
// 높아야 유효. 미만이면 다중경로/잡음으로 보고 그 프레임을 버린다(9999 반환).
#define PEAK_SIDELOBE_RATIO 1.5f
#define PSR_GUARD           3
// PHAT 백색화 지수: 1.0 = 완전 PHAT(가장 뾰족하나 잡음 민감), 0 = 순수 상호상관.
// 0.7이면 선명도 대부분을 유지하면서 잡음-only 빈의 과증폭을 억제(반향에 강함).
#define PHAT_BETA           0.7f
// EMA 앞단 중앙값 필터 길이(홀수). 단일 프레임 튐 제거용.
#define MED_LEN             3
// 💡 0에 가까울수록 부드럽지만(잡음에 강하지만) 소리가 움직일 때 반응이 느려지고,
// 1에 가까울수록 반응은 빠르지만 잡음(떨림)이 그대로 각도에 드러납니다.
#define DELAY_EMA_ALPHA     0.35f
#define USE_BANDPASS        true
#define MIN_FREQ            200.0f
#define MAX_FREQ            5000.0f
// 💡 보드가 몸에 장착된 각도 때문에 실제 방향과 표시 각도가 일정하게 어긋난다면
// (예: 실제로는 정면인데 항상 15도쯤 오른쪽으로 나온다면) 이 값을 조정하세요.
// 표시 각도 = 계산된 각도 + 이 값 (도 단위, 음수도 가능).
#define ANGLE_CALIBRATION_OFFSET_DEG 0.0f

int32_t raw_buf_i2s0[SAMPLES_PER_READ * 2];
int32_t raw_buf_i2s1[SAMPLES_PER_READ * 2];

float buf_E[SAMPLES_PER_READ], buf_W[SAMPLES_PER_READ];
float buf_S[SAMPLES_PER_READ], buf_N[SAMPLES_PER_READ];

float vReal1[SAMPLES_PER_READ], vImag1[SAMPLES_PER_READ];
float vReal2[SAMPLES_PER_READ], vImag2[SAMPLES_PER_READ];

ArduinoFFT FFT1 = ArduinoFFT(vReal1, vImag1, SAMPLES_PER_READ, FS);
ArduinoFFT FFT2 = ArduinoFFT(vReal2, vImag2, SAMPLES_PER_READ, FS);

// 예전에는 최근 8개 표본을 단순평균했는데(FILTER_SIZE), 그러면 창(윈도우) 절반 크기만큼
// (~250-400ms) 지연이 생겨서 소리가 움직일 때 표시 각도가 실제보다 뒤처졌습니다.
// 지수이동평균(EMA)은 같은 수준의 잡음 억제력에서 지연이 훨씬 짧습니다.
float ema_delay_x = 0.0f, ema_delay_y = 0.0f;
bool ema_initialized = false;

// EMA 앞단 중앙값 필터용 링버퍼 (raw 지연 최근 MED_LEN개)
float dx_hist[MED_LEN] = {0}, dy_hist[MED_LEN] = {0};
int med_count = 0, med_idx = 0;

// 최신으로 계산된 각도만 들고 있다가, 메인(TinyML) 보드가 GET_ANGLE을 요청할 때만 응답한다.
// (예전 버전처럼 무조건 브로드캐스트하면 "판단 시점"과 무관한 각도가 붙어버림 -
//  main.cpp의 request_angle_sync()가 기대하는 프로토콜이 바로 이 요청/응답 방식이다.)
// -1.0은 "아직 방향 계산 안 됨" 표시. 0.0으로 두면 진짜 유효한 방향(정북)과 구분이
// 안 돼서 main.cpp가 "아직 모름"을 "정북에서 소리남"으로 착각하게 된다. 범위(0~360)
// 밖의 -1.0은 main.cpp의 parse_angle_line() 범위 체크에서 자동으로 거부되어
// GET_ANGLE 타임아웃과 동일하게 처리됨(별도 분기 불필요).
float last_angle_deg = -1.0f;
char rx_line[16] = {0};
size_t rx_used = 0;

// 메인 보드로부터 UART(Serial1)로 "SLEEP" 수신 시 딥슬립 진입.
// GPIO12를 다시 LOW로 만들면(버튼) 깨어나 setup()부터 재시작한다.
void enter_deep_sleep() {
  Serial.println("🌙 딥슬립 진입");
  digitalWrite(LED_PIN, LOW);
  i2s_driver_uninstall(I2S_NUM_0);
  i2s_driver_uninstall(I2S_NUM_1);
  // 딥슬립 중 웨이크 핀 플로팅 방지 (UART 등 노이즈로 즉시 깨는 현상 차단)
  rtc_gpio_pullup_en((gpio_num_t)BUTTON_PIN);
  rtc_gpio_pulldown_dis((gpio_num_t)BUTTON_PIN);
  esp_sleep_enable_ext0_wakeup((gpio_num_t)BUTTON_PIN, 0);  // LOW에서 기상
  esp_deep_sleep_start();
}

void check_angle_request() {
  while (Serial1.available()) {
    char c = (char)Serial1.read();
    if (c == '\n') {
      rx_line[rx_used] = '\0';
      if (strncmp(rx_line, "GET_ANGLE", 9) == 0) {
        Serial1.println(last_angle_deg);
      } else if (strcmp(rx_line, "SLEEP") == 0) {
        enter_deep_sleep();
      }
      rx_used = 0;
      continue;
    }
    if (c == '\r') continue;
    if (rx_used < sizeof(rx_line) - 1) rx_line[rx_used++] = c;
  }
}

float calculate_energy(const float* buffer, int len) {
  double sum = 0.0;
  for (int i = 0; i < len; ++i) sum += (double)buffer[i] * buffer[i];
  return (float)(sum / len);
}

// 유효하지 않은 프레임을 나타내는 값 (max_delay 범위 밖이라 loop에서 걸러진다).
#define GCC_INVALID 9999.0f

// GCC-PHAT 지연을 소수점(서브샘플) 정밀도로 반환한다.
float compute_gcc_phat_delay(float* sig1, float* sig2) {
  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    vReal1[i] = sig1[i]; vImag1[i] = 0;
    vReal2[i] = sig2[i]; vImag2[i] = 0;
  }
  FFT1.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
  FFT1.compute(FFT_FORWARD);
  FFT2.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
  FFT2.compute(FFT_FORWARD);

  if (USE_BANDPASS) {
    float freq_res = FS / SAMPLES_PER_READ;
    int min_bin = (int)(MIN_FREQ / freq_res);
    int max_bin = (int)(MAX_FREQ / freq_res);
    for (int i = 1; i < (SAMPLES_PER_READ / 2); i++) {
      if (i < min_bin || i > max_bin) {
        vReal1[i] = 0; vImag1[i] = 0; vReal2[i] = 0; vImag2[i] = 0;
        vReal1[SAMPLES_PER_READ - i] = 0; vImag1[SAMPLES_PER_READ - i] = 0;
        vReal2[SAMPLES_PER_READ - i] = 0; vImag2[SAMPLES_PER_READ - i] = 0;
      }
    }
  }

  // 교차 스펙트럼에 PHAT-β 가중치: r*|R|^(-beta). beta<1이면 잡음-only 빈의
  // 진폭이 덜 부풀려져 실내 반향/저SNR에서 피크가 더 안정적이다.
  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    float r = vReal1[i] * vReal2[i] + vImag1[i] * vImag2[i];
    float im = vImag1[i] * vReal2[i] - vReal1[i] * vImag2[i];
    float mag = sqrtf(r * r + im * im);
    if (mag > 1.0e-12f) {
      float w = 1.0f / powf(mag, PHAT_BETA);
      vReal1[i] = r * w; vImag1[i] = im * w;
    } else {
      vReal1[i] = 0; vImag1[i] = 0;
    }
  }
  FFT1.compute(FFT_REVERSE);

  // 메인 피크
  float max_corr = -1.0e30f;
  int peak = 0;
  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    if (vReal1[i] > max_corr) { max_corr = vReal1[i]; peak = i; }
  }
  if (max_corr <= 0.0f) return GCC_INVALID;

  // 피크에서 PSR_GUARD 샘플 이상(원형 거리) 떨어진 최대 부엽
  float sidelobe = -1.0e30f;
  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    int d = abs(i - peak);
    if (d > SAMPLES_PER_READ / 2) d = SAMPLES_PER_READ - d;
    if (d >= PSR_GUARD && vReal1[i] > sidelobe) sidelobe = vReal1[i];
  }
  if (sidelobe > 0.0f && max_corr < sidelobe * PEAK_SIDELOBE_RATIO) return GCC_INVALID;

  // 서브샘플 보간: 피크 주변 3점 포물선 피팅 (원형 이웃)
  int pm = (peak - 1 + SAMPLES_PER_READ) % SAMPLES_PER_READ;
  int pp = (peak + 1) % SAMPLES_PER_READ;
  float ym1 = vReal1[pm], y0 = vReal1[peak], yp1 = vReal1[pp];
  float denom = ym1 - 2.0f * y0 + yp1;
  float frac = 0.0f;
  if (fabsf(denom) > 1.0e-12f) {
    frac = 0.5f * (ym1 - yp1) / denom;
    if (frac > 1.0f) frac = 1.0f; else if (frac < -1.0f) frac = -1.0f;
  }

  float delay = (float)peak + frac;
  if (delay >= SAMPLES_PER_READ / 2) delay -= SAMPLES_PER_READ;
  return delay;
}

// 최대 MED_LEN개 값의 중앙값 (n <= MED_LEN, 삽입정렬).
float median_of(const float* src, int n) {
  float t[MED_LEN];
  for (int i = 0; i < n; i++) t[i] = src[i];
  for (int i = 1; i < n; i++) {
    float v = t[i]; int j = i - 1;
    while (j >= 0 && t[j] > v) { t[j + 1] = t[j]; j--; }
    t[j + 1] = v;
  }
  return t[n / 2];
}

void setup() {
  Serial.begin(115200); // 디버깅용 PC 연결
  Serial1.begin(115200, SERIAL_8N1, UART_RX, UART_TX); // 보드 간 통신용 설정

  // LED 설정 및 켜기 (깨어있음을 표시)
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  // 버튼 핀 설정 + 웨이크업 시 누르고 있던 버튼에서 손을 뗄 때까지 대기
  // (루프 진입 후 곧바로 다시 딥슬립에 빠지는 현상 방지)
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  while (digitalRead(BUTTON_PIN) == LOW) { delay(10); }
  delay(50);

  i2s_config_t i2s_cfg = {};
  i2s_cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  i2s_cfg.sample_rate = (uint32_t)FS;
  i2s_cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  i2s_cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  i2s_cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  i2s_cfg.intr_alloc_flags = 0;
  i2s_cfg.dma_buf_count = 8;
  i2s_cfg.dma_buf_len = SAMPLES_PER_READ;
  i2s_cfg.use_apll = false;

  i2s_pin_config_t pins0 = { .bck_io_num = I2S0_SCK, .ws_io_num = I2S0_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S0_SD };
  i2s_driver_install(I2S_NUM_0, &i2s_cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins0);
  i2s_zero_dma_buffer(I2S_NUM_0);

  i2s_pin_config_t pins1 = { .bck_io_num = I2S1_SCK, .ws_io_num = I2S1_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S1_SD };
  i2s_driver_install(I2S_NUM_1, &i2s_cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_1, &pins1);
  i2s_zero_dma_buffer(I2S_NUM_1);

  Serial.println("보드 1: TDOA 방향 추정 시작...");
}

void loop() {
  // 물리적 버튼으로도 직접 딥슬립 진입 (메인 보드의 UART "SLEEP" 명령과는 별개 경로)
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(50); // 디바운스
    if (digitalRead(BUTTON_PIN) == LOW) {
      Serial.println("🌙 물리적 버튼 입력으로 딥슬립에 진입합니다...");
      while (digitalRead(BUTTON_PIN) == LOW) { delay(10); } // 버튼에서 손을 뗄 때까지 대기
      delay(50);
      enter_deep_sleep();
    }
  }

  check_angle_request();

  size_t bytes_read0 = 0, bytes_read1 = 0;
  i2s_read(I2S_NUM_0, raw_buf_i2s0, sizeof(raw_buf_i2s0), &bytes_read0, portMAX_DELAY);
  i2s_read(I2S_NUM_1, raw_buf_i2s1, sizeof(raw_buf_i2s1), &bytes_read1, portMAX_DELAY);

  for (int i = 0; i < SAMPLES_PER_READ; ++i) {
    buf_E[i] = (float)(raw_buf_i2s0[2 * i + 0] >> 8) / 8388608.0f;
    buf_W[i] = (float)(raw_buf_i2s0[2 * i + 1] >> 8) / 8388608.0f;
    buf_N[i] = (float)(raw_buf_i2s1[2 * i + 0] >> 8) / 8388608.0f;
    buf_S[i] = (float)(raw_buf_i2s1[2 * i + 1] >> 8) / 8388608.0f;
  }

  float eE = calculate_energy(buf_E, SAMPLES_PER_READ);
  float eW = calculate_energy(buf_W, SAMPLES_PER_READ);
  float eS = calculate_energy(buf_S, SAMPLES_PER_READ);
  float eN = calculate_energy(buf_N, SAMPLES_PER_READ);

  // 설정된 임계치(ENERGY_THRESH)보다 소리가 작으면 무시 (노이즈 필터링)
  if (eE < ENERGY_THRESH && eW < ENERGY_THRESH && eS < ENERGY_THRESH && eN < ENERGY_THRESH) return;

  // 💡 만약 방향이 반대로(거울처럼) 나온다면 아래 인자 순서를 (buf_W, buf_E) 및 (buf_S, buf_N)으로 바꾸세요.
  float delay_x = compute_gcc_phat_delay(buf_E, buf_W);
  float delay_y = compute_gcc_phat_delay(buf_N, buf_S);

  float max_delay = ceilf((DISTANCE_MICS_M / SOUND_SPEED_MPS) * FS);
  if (fabsf(delay_x) > max_delay + 2.0f || fabsf(delay_y) > max_delay + 2.0f) return;

  // 중앙값 필터: 최근 MED_LEN개 raw 지연의 중앙값 -> 단일 프레임 튐 제거
  dx_hist[med_idx] = delay_x;
  dy_hist[med_idx] = delay_y;
  med_idx = (med_idx + 1) % MED_LEN;
  if (med_count < MED_LEN) med_count++;
  float med_x = median_of(dx_hist, med_count);
  float med_y = median_of(dy_hist, med_count);

  if (!ema_initialized) {
    ema_delay_x = med_x;
    ema_delay_y = med_y;
    ema_initialized = true;
  } else {
    ema_delay_x += DELAY_EMA_ALPHA * (med_x - ema_delay_x);
    ema_delay_y += DELAY_EMA_ALPHA * (med_y - ema_delay_y);
  }

  // 💡 [수정됨] 북쪽(N)을 0도, 동쪽(E)을 90도로 설정하기 위해 atan2f의 인자 순서를 (ema_delay_x, ema_delay_y)로 변경
  float angle_deg = atan2f(ema_delay_x, ema_delay_y) * (180.0f / M_PI);
  angle_deg += ANGLE_CALIBRATION_OFFSET_DEG;
  angle_deg = fmodf(angle_deg, 360.0f);
  if (angle_deg < 0.0f) angle_deg += 360.0f;

  // 무조건 전송하지 않고 최신 각도만 보관 -> 메인 보드가 GET_ANGLE로 요청할 때 응답
  last_angle_deg = angle_deg;

  // 시리얼 모니터 확인용 출력
  Serial.print("계산된 각도: "); Serial.println(angle_deg);
}
