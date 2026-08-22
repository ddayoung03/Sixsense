#include <Arduino.h>
#include <math.h>
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
#define FILTER_SIZE         8         
#define USE_BANDPASS        true
#define MIN_FREQ            200.0f
#define MAX_FREQ            5000.0f

int32_t raw_buf_i2s0[SAMPLES_PER_READ * 2]; 
int32_t raw_buf_i2s1[SAMPLES_PER_READ * 2]; 

float buf_E[SAMPLES_PER_READ], buf_W[SAMPLES_PER_READ]; 
float buf_S[SAMPLES_PER_READ], buf_N[SAMPLES_PER_READ]; 

float vReal1[SAMPLES_PER_READ], vImag1[SAMPLES_PER_READ];
float vReal2[SAMPLES_PER_READ], vImag2[SAMPLES_PER_READ];

ArduinoFFT FFT1 = ArduinoFFT(vReal1, vImag1, SAMPLES_PER_READ, FS);
ArduinoFFT FFT2 = ArduinoFFT(vReal2, vImag2, SAMPLES_PER_READ, FS);

float x_delay_history[FILTER_SIZE] = {0};
float y_delay_history[FILTER_SIZE] = {0};
int filter_idx = 0;

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
    corr_sum += vReal1[i];
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

  x_delay_history[filter_idx] = (float)delay_x;
  y_delay_history[filter_idx] = (float)delay_y;
  filter_idx = (filter_idx + 1) % FILTER_SIZE;

  float avg_x = 0.0f, avg_y = 0.0f;
  for (int i = 0; i < FILTER_SIZE; i++) { avg_x += x_delay_history[i]; avg_y += y_delay_history[i]; }
  avg_x /= FILTER_SIZE; avg_y /= FILTER_SIZE;

  // 💡 [수정됨] 북쪽(N)을 0도, 동쪽(E)을 90도로 설정하기 위해 atan2f의 인자 순서를 (avg_x, avg_y)로 변경
  float angle_deg = atan2f(avg_x, avg_y) * (180.0f / M_PI);
  if (angle_deg < 0.0f) angle_deg += 360.0f;

  // 2번 보드(TinyML 보드)로 계산된 각도를 전송
  Serial1.println(angle_deg);
  
  // 시리얼 모니터 확인용 출력
  Serial.print("전송된 각도: "); Serial.println(angle_deg);
}