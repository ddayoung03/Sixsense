#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "driver/i2s.h"
#include "arduinoFFT.h"

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
#define CORR_QUALITY_THRESH 1.8f
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

// 최신으로 계산된 각도만 들고 있다가, 2번 보드(TinyML)가 GET_ANGLE을 요청할 때만 응답한다.
// (예전에는 무조건 브로드캐스트했는데, 그러면 "판단 시점"과 무관한 각도가 붙어버림)
float last_angle_deg = 0.0f;
char rx_line[16] = {0};
size_t rx_used = 0;

void check_angle_request() {
  while (Serial1.available()) {
    char c = (char)Serial1.read();
    if (c == '\n') {
      rx_line[rx_used] = '\0';
      if (strncmp(rx_line, "GET_ANGLE", 9) == 0) {
        Serial1.println(last_angle_deg);
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

int compute_gcc_phat_delay(float* sig1, float* sig2) {
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
  double corr_sum = 0;

  for (int i = 0; i < SAMPLES_PER_READ; i++) {
    // 절대값 합산: 그냥 더하면 +/-가 상쇄되어 avg_corr가 0 근처(혹은 음수)가 되고,
    // 그러면 아래 품질 문턱값이 사실상 무력화되어 노이즈에도 각도가 튀는 원인이 된다.
    corr_sum += fabsf(vReal1[i]);
    if (vReal1[i] > max_corr) { max_corr = vReal1[i]; delay_index = i; }
  }

  float avg_corr = (float)(corr_sum / SAMPLES_PER_READ);
  if (max_corr < avg_corr * CORR_QUALITY_THRESH) return 9999; 

  if (delay_index >= SAMPLES_PER_READ / 2) delay_index -= SAMPLES_PER_READ;
  return delay_index;
}

void setup() {
  Serial.begin(115200); // 디버깅용 PC 연결
  Serial1.begin(115200, SERIAL_8N1, UART_RX, UART_TX); // 보드 간 통신용 설정
  
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
  int delay_x = compute_gcc_phat_delay(buf_E, buf_W);
  int delay_y = compute_gcc_phat_delay(buf_N, buf_S);

  int max_delay = (int)ceilf((DISTANCE_MICS_M / SOUND_SPEED_MPS) * FS);
  if (abs(delay_x) > max_delay + 2 || abs(delay_y) > max_delay + 2) return;

  if (!ema_initialized) {
    ema_delay_x = (float)delay_x;
    ema_delay_y = (float)delay_y;
    ema_initialized = true;
  } else {
    ema_delay_x += DELAY_EMA_ALPHA * ((float)delay_x - ema_delay_x);
    ema_delay_y += DELAY_EMA_ALPHA * ((float)delay_y - ema_delay_y);
  }

  // 💡 [수정됨] 북쪽(N)을 0도, 동쪽(E)을 90도로 설정하기 위해 atan2f의 인자 순서를 (ema_delay_x, ema_delay_y)로 변경
  float angle_deg = atan2f(ema_delay_x, ema_delay_y) * (180.0f / M_PI);
  angle_deg += ANGLE_CALIBRATION_OFFSET_DEG;
  angle_deg = fmodf(angle_deg, 360.0f);
  if (angle_deg < 0.0f) angle_deg += 360.0f;

  // 무조건 전송하지 않고 최신 각도만 보관 -> 2번 보드가 GET_ANGLE로 요청할 때 응답
  last_angle_deg = angle_deg;

  // 시리얼 모니터 확인용 출력
  Serial.print("계산된 각도: "); Serial.println(angle_deg);
}