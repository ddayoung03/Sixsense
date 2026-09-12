#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "driver/i2s.h"
#include "driver/rtc_io.h"
#include "arduinoFFT.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ===== 듀얼코어 분담 =====
// Core 0: compute_task - I2S 읽기 + FFT/GCC-PHAT만 무한 반복, 계산된 최신
//         각도를 last_angle_deg에 계속 덮어쓴다. 딥슬립도 얘가 실행한다
//         (I2S 드라이버를 얘가 소유하므로 - 다른 태스크가 얘 i2s_read 도중에
//         드라이버를 내리면 위험).
// Core 1: loop() (Arduino 기본 태스크) - UART(GET_ANGLE/SLEEP)와 버튼만
//         본다. 계산이 얼마나 걸리든 신경 안 쓰고 last_angle_deg를 즉시
//         읽어서 응답 - 예전엔 이 응답이 매 루프의 FFT 계산 뒤에야 처리돼서
//         최악의 경우 한 프레임(수십ms) 지연이 있었다.
// last_angle_deg는 두 코어가 같이 건드리므로 portMUX 스핀락으로 보호한다.
static portMUX_TYPE angle_mux = portMUX_INITIALIZER_UNLOCKED;

// 딥슬립 요청 플래그: core1(loop)이 세팅하고, core0(compute_task)가 자기
// i2s_read 사이사이 안전한 지점에서 확인 후 직접 딥슬립에 들어간다.
// 단일 writer(loop)/단일 reader(compute_task) 조합이라 volatile bool만으로 충분.
static volatile bool sleep_requested = false;

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
// 💡 반응속도 최우선 모드: 정확도/떨림은 포기하고 지연을 최소화한다.
// - SAMPLES_PER_READ를 줄여서 마이크 버퍼 채우는 시간 자체를 줄임(1024→256, 64ms→16ms).
// - 이동평균 필터와 상관 품질 게이트를 모두 제거해서 매 프레임 값을 그대로 즉시 내보냄.
#define FS                  16000.0f
#define SAMPLES_PER_READ    256
#define DISTANCE_MICS_M     0.25f
#define SOUND_SPEED_MPS     343.0f
// 💡 만약 너무 작은 소리에도 반응해서 각도가 튄다면 이 값을 키우고(예: 1.0e-5f),
// 소리를 잘 못 잡으면 이 값을 줄여주세요(예: 1.0e-6f).
#define ENERGY_THRESH       5.0e-6f
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

// 최신으로 계산된 각도(각도_mux로 보호). 메인(TinyML) 보드가 GET_ANGLE을
// 요청할 때 loop()가 이 값을 그대로 응답한다 - compute_task의 계산 시간과
// 무관하게 즉시 응답 가능한 이유가 이 분리 구조.
// -1.0은 "아직 방향 계산 안 됨" 표시. 0.0으로 두면 진짜 유효한 방향(정북)과 구분이
// 안 돼서 main.cpp가 "아직 모름"을 "정북에서 소리남"으로 착각하게 된다. 범위(0~360)
// 밖의 -1.0은 main.cpp의 parse_angle_line() 범위 체크에서 자동으로 거부되어
// GET_ANGLE 타임아웃과 동일하게 처리됨(별도 분기 불필요).
float last_angle_deg = -1.0f;
char rx_line[16] = {0};
size_t rx_used = 0;

// Core0(compute_task) 전용: I2S 드라이버를 소유한 태스크가 직접 정리하고
// 잠든다. GPIO12를 다시 LOW로 만들면(버튼) 깨어나 setup()부터 재시작한다.
void enter_deep_sleep() {
  Serial.println("🌙 딥슬립 진입 (compute_task, core0)");
  digitalWrite(LED_PIN, LOW);
  i2s_driver_uninstall(I2S_NUM_0);
  i2s_driver_uninstall(I2S_NUM_1);
  // 딥슬립 중 웨이크 핀 플로팅 방지 (UART 등 노이즈로 즉시 깨는 현상 차단)
  rtc_gpio_pullup_en((gpio_num_t)BUTTON_PIN);
  rtc_gpio_pulldown_dis((gpio_num_t)BUTTON_PIN);
  esp_sleep_enable_ext0_wakeup((gpio_num_t)BUTTON_PIN, 0);  // LOW에서 기상
  esp_deep_sleep_start();
}

float calculate_energy(const float* buffer, int len) {
  double sum = 0.0;
  for (int i = 0; i < len; ++i) sum += (double)buffer[i] * buffer[i];
  return (float)(sum / len);
}

// GCC-PHAT 지연을 정수 샘플 정밀도로 반환한다 (구버전 방식).
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

  // 순수 PHAT 백색화: r/|r| (구버전 방식, beta 감쇠 없음 - 가장 뾰족한 피크지만 잡음에 민감)
  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    float r = vReal1[i] * vReal2[i] + vImag1[i] * vImag2[i];
    float im = vImag1[i] * vReal2[i] - vReal1[i] * vImag2[i];
    float mag = sqrtf(r * r + im * im);
    if (mag > 1.0e-9f) {
      vReal1[i] = r / mag; vImag1[i] = im / mag;
    } else {
      vReal1[i] = 0; vImag1[i] = 0;
    }
  }
  FFT1.compute(FFT_REVERSE);

  float max_corr = -1000.0f;
  int delay_index = 0;

  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    if (vReal1[i] > max_corr) { max_corr = vReal1[i]; delay_index = i; }
  }

  if (delay_index >= SAMPLES_PER_READ / 2) delay_index -= SAMPLES_PER_READ;
  return (float)delay_index;
}

// ===== Task A (core0): I2S 읽기 + GCC-PHAT 계산 전담 =====
// UART/버튼은 절대 건드리지 않는다. 오직 마이크 읽고 각도 계산해서
// last_angle_deg에 덮어쓰는 것만 무한 반복.
void compute_task(void* pvParameters) {
  for (;;) {
    // 딥슬립 요청은 i2s_read 시작 전, 안전한 시점에만 처리 (읽기 도중 드라이버
    // 내리는 일이 없도록).
    if (sleep_requested) {
      enter_deep_sleep();  // 반환하지 않음
    }

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
    if (eE < ENERGY_THRESH && eW < ENERGY_THRESH && eS < ENERGY_THRESH && eN < ENERGY_THRESH) continue;

    // 💡 만약 방향이 반대로(거울처럼) 나온다면 아래 인자 순서를 (buf_W, buf_E) 및 (buf_S, buf_N)으로 바꾸세요.
    float delay_x = compute_gcc_phat_delay(buf_E, buf_W);
    float delay_y = compute_gcc_phat_delay(buf_N, buf_S);

    float max_delay = ceilf((DISTANCE_MICS_M / SOUND_SPEED_MPS) * FS);
    if (fabsf(delay_x) > max_delay + 2.0f || fabsf(delay_y) > max_delay + 2.0f) continue;

    // 평균/스무딩 없이 이번 프레임 값을 그대로 즉시 사용 (반응속도 최우선, 떨림은 감수)
    // 북쪽(N)을 0도, 동쪽(E)을 90도로 설정하기 위해 atan2f의 인자 순서를 (delay_x, delay_y)로 유지
    float angle_deg = atan2f(delay_x, delay_y) * (180.0f / M_PI);
    angle_deg += ANGLE_CALIBRATION_OFFSET_DEG;
    angle_deg = fmodf(angle_deg, 360.0f);
    if (angle_deg < 0.0f) angle_deg += 360.0f;

    portENTER_CRITICAL(&angle_mux);
    last_angle_deg = angle_deg;
    portEXIT_CRITICAL(&angle_mux);

    // 시리얼 모니터 확인용 출력
    Serial.print("계산된 각도: "); Serial.println(angle_deg);
  }
}

void setup() {
  Serial.begin(115200); // 디버깅용 PC 연결
  // USB CDC는 호스트가 연결돼 있어도 수신 쪽이 느리면 Serial.print가 최대
  // 250ms(기본 tx_timeout_ms)까지 블로킹된다. compute_task(core0)가 매
  // 16ms(SAMPLES_PER_READ=256 기준)마다 디버그 출력을 하는 지금 구조에서는
  // 이 블로킹 한 번이 반응속도 최적화 전체를 무의미하게 만들 수 있으므로,
  // 타임아웃을 0으로 두어 보낼 자리가 없으면 그냥 버리고 즉시 리턴하게 한다.
  Serial.setTxTimeoutMs(0);
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
  // 반응속도 우선: DMA에 쌓아두는 버퍼 수를 줄여 오래된 오디오를 읽는 지연을 최소화
  i2s_cfg.dma_buf_count = 3;
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

  Serial.println("보드 1: TDOA 방향 추정 시작 (듀얼코어: core0=계산, core1=통신)...");

  // Task A를 core0에 고정. Arduino의 기본 loop()/setup() 태스크(Task B 역할)는
  // 이미 core1에서 돌고 있으므로 별도로 만들 필요 없음.
  xTaskCreatePinnedToCore(compute_task, "compute_task", 8192, nullptr, 1, nullptr, 0);
}

// ===== Task B (core1, Arduino 기본 루프): 통신 전담 =====
// 계산은 전혀 하지 않는다. UART 수신함과 버튼만 계속 확인 - compute_task가
// 지금 계산 중이든 아니든 상관없이 last_angle_deg를 즉시 읽어서 응답한다.
void loop() {
  // 물리적 버튼: 여기선 플래그만 세팅. 실제 딥슬립 진입은 I2S를 소유한
  // compute_task(core0)가 안전한 시점에 직접 수행한다.
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(50); // 디바운스
    if (digitalRead(BUTTON_PIN) == LOW) {
      Serial.println("🌙 물리적 버튼 입력 - 딥슬립 요청 (core0에 위임)");
      while (digitalRead(BUTTON_PIN) == LOW) { delay(10); } // 버튼에서 손을 뗄 때까지 대기
      delay(50);
      sleep_requested = true;
    }
  }

  while (Serial1.available()) {
    char c = (char)Serial1.read();
    if (c == '\n') {
      rx_line[rx_used] = '\0';
      if (strncmp(rx_line, "GET_ANGLE", 9) == 0) {
        portENTER_CRITICAL(&angle_mux);
        float angle = last_angle_deg;
        portEXIT_CRITICAL(&angle_mux);
        Serial1.println(angle);  // compute_task 계산 시간과 무관하게 즉시 응답
      } else if (strcmp(rx_line, "SLEEP") == 0) {
        sleep_requested = true;
      }
      rx_used = 0;
      continue;
    }
    if (c == '\r') continue;
    if (rx_used < sizeof(rx_line) - 1) rx_line[rx_used++] = c;
  }

  delay(1); // 워치독 피하면서 최대한 촘촘히 폴링 (응답 지연 ~1ms 목표)
}
