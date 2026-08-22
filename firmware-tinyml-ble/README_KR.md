# GeekbleSound ESP-IDF v3.2 RMS FINAL

업로드된 `horn_noise_siren_model_v3_2_rms.zip`을 기준으로 만든 ESP-IDF 프로젝트입니다.

## 런타임 파이프라인

1. INMP441에서 16 kHz mono, 2초(32,000 sample) 수집
2. raw 32-bit I2S slot을 `raw >> 16`으로 PCM16 변환
3. DC offset 제거 기준 RMS 계산
4. `RMS < 150`이면 추론 생략
5. 활성 입력을 RMS 2500 근처로 정규화
   - gain = clamp(2500 / RMS, 0.25, 8.0)
   - peak가 float 0.98을 넘으면 limiter 적용
6. librosa 호환 Log-Mel: FFT 512 / hop 256 / mel 64 / 50~8000 Hz / Slaney / top_db 80
7. dB -80~20 clamp
8. 새 v3.2 TFLite 입력 양자화(`scale=0.371063441`, `zero=79`)
9. TFLite 추론: horn / noise / siren
10. 최고 score가 0.75 미만이면 `uncertain`

## 배선

- INMP441 VDD -> 3.3V
- INMP441 GND -> GND
- INMP441 SCK -> Geekble D3 / GPIO 6
- INMP441 WS  -> Geekble D4 / GPIO 7
- INMP441 SD  -> Geekble D2 / GPIO 5
- INMP441 L/R -> GND (LEFT)

## VS Code / ESP-IDF 5.4.4

프로젝트 폴더 자체를 VS Code에서 엽니다.

터미널에서 최초 1회:

```powershell
idf.py set-target esp32s3
idf.py reconfigure
idf.py build
```

그 다음 보드 연결 후:

```powershell
idf.py flash monitor
```

또는 VS Code ESP-IDF 확장의 Build / Flash / Monitor 버튼을 사용합니다.

### Component

`main/idf_component.yml`에 `espressif/esp-tflite-micro 1.3.5`를 고정했습니다. 최초 configure/build 때 Component Manager가 내려받습니다. `partitions.csv`는 4MB Flash에서 앱 영역을 3MB로 잡아 TFLite Micro 코드 공간을 확보합니다.

## PSRAM

`sdkconfig.defaults`에 Quad/QSPI PSRAM 설정을 넣었습니다. Tensor Arena, PCM, Log-Mel 버퍼는 PSRAM을 우선 사용하고 실패하면 internal RAM을 시도합니다.

## 중요 변경점

이전 모델 전용 `real_sample_test.h` 부팅 검사는 제거했습니다. 새 v3.2 모델과 이전 reference output은 서로 다른 모델이므로 그대로 비교하면 정상 모델도 실패합니다. 대신 부팅 시 새 모델의 tensor type/shape/quantization을 `model_info.json`과 대조하고 일치하지 않으면 중단합니다.

## 원본 모델

`model_original/`에 원본 `.tflite`, `model_info.json`, 학습 기록과 내부 테스트 CSV를 보존했습니다. 실제 펌웨어는 `main/model_data.h`의 12,192-byte INT8 모델을 사용합니다.
