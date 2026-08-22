# Sixsense — 위험 소리 방향 인식 웨어러블

청각장애인을 위한 위험 소리(경적/사이렌) 인식 및 방향 알림 웨어러블 시스템. 서로 통신하는 3개의 서브프로젝트로 구성됩니다.

## 구성

```
watch-app/             Wear OS 워치 앱 (Kotlin, Jetpack Compose)
                        - ESP32에서 BLE로 "소리종류,각도"를 수신해 화면에 방향/아이콘 표시
                        - 소리 종류(경적/사이렌)별 진동 패턴 + 사용자 진동 세기 설정

firmware-tdoa/          방향추정 보드 (Arduino, ESP32)
                        - 4채널 마이크(전/후/좌/우)로 GCC-PHAT 기반 TDOA 계산
                        - 산출된 각도(0~360도)를 UART로 TinyML 보드에 전달

firmware-tinyml-ble/    음향 분류 + BLE 보드 (ESP-IDF, ESP32-S3)
                        - INMP441 마이크로 2초 단위 오디오 캡처
                        - Log-Mel Spectrogram 추출 → INT8 TFLite Micro 모델로 horn/noise/siren 분류
                        - UART로 각도 수신, BLE(NimBLE)로 워치에 "소리종류,각도" 전송
```

## 빌드 방법

### watch-app
```
cd watch-app
./gradlew assembleDebug
```
Wear OS 기기(또는 에뮬레이터)에 설치 후 실행.

### firmware-tdoa
Arduino IDE에서 `tdoa_connect.ino` 오픈 → 보드를 ESP32(긱블 나노)로 선택 → 업로드.
`arduinoFFT` 라이브러리 필요.

### firmware-tinyml-ble
ESP-IDF(v5.x) 환경에서:
```
cd firmware-tinyml-ble
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```
`main/idf_component.yml`에 명시된 컴포넌트(`espressif__esp-tflite-micro` 등)는 빌드 시 자동으로 받아옵니다.

## 하드웨어 연결 개요

```
[4채널 마이크] → firmware-tdoa 보드 → (UART, 각도) → firmware-tinyml-ble 보드 → (BLE) → watch-app
                                                              ↑
                                                        [단일 마이크]
```
