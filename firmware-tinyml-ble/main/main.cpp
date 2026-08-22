#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <new>

#include "driver/i2s_std.h"
#include "driver/uart.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// BLE (ESP-IDF NimBLE)
#include "esp_mac.h"
#include "nvs_flash.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "fft512.h"
#include "labels.h"
#include "mel_filterbank.h"
#include "model_data.h"

namespace {
constexpr char TAG[] = "GEEKBLE_AI";

// -----------------------------------------------------------------------------
// BLE: same Service / Characteristic UUIDs as BLE_connect_namming.ino.
// Payload format sent to the watch: "angle,horn" or "angle,siren".
// Angle is received from another ESP32 over UART. Until the first valid angle
// arrives, 90 degrees is used as a safe default.
// -----------------------------------------------------------------------------
constexpr float DEFAULT_ANGLE_DEG = 90.0f;

// -----------------------------------------------------------------------------
// Angle UART link - matched to tinyml_BLE.ino Serial1.begin(115200, 8N1, RX, TX)
// on Geekble nano ESP32-S3.
//
// Geekble nano pin map:
//   D0/RX = GPIO44  <- connect the OTHER ESP32 TX here
//   D1/TX = GPIO43  -> connect to the OTHER ESP32 RX only if bidirectional UART
//   GND   = GND
//
// Protocol: ASCII floating-point angle followed by newline.
// Examples: "0\n", "45.5\n", "90.0\n", "135\n"
// -----------------------------------------------------------------------------
constexpr uart_port_t ANGLE_UART_PORT = UART_NUM_1;
constexpr int ANGLE_UART_BAUD = 115200;
constexpr int ANGLE_UART_RX_PIN = 44;  // Geekble D0/RX
constexpr int ANGLE_UART_TX_PIN = 43;  // Geekble D1/TX
constexpr float ANGLE_MIN_DEG = 0.0f;
constexpr float ANGLE_MAX_DEG = 360.0f;
constexpr char BLE_SERVICE_UUID_TEXT[] = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
constexpr char BLE_CHARACTERISTIC_UUID_TEXT[] = "beb5483e-36e1-4688-b7f5-ea07361b26a8";

// NimBLE stores 128-bit UUIDs least-significant byte first.
static const ble_uuid128_t BLE_SERVICE_UUID = BLE_UUID128_INIT(
    0x4b, 0x91, 0x31, 0xc3, 0xc9, 0xc5, 0xcc, 0x8f,
    0x9e, 0x45, 0xb5, 0x1f, 0x01, 0xc2, 0xaf, 0x4f);
static const ble_uuid128_t BLE_CHARACTERISTIC_UUID = BLE_UUID128_INIT(
    0xa8, 0x26, 0x1b, 0x36, 0x07, 0xea, 0xf5, 0xb7,
    0x88, 0x46, 0xe1, 0x36, 0x3e, 0x48, 0xb5, 0xbe);

static uint8_t ble_addr_type = 0;
static uint16_t ble_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint16_t ble_char_val_handle = 0;
static volatile bool ble_notify_enabled = false;
static char ble_device_name[40] = "ESP32_AUDIO_ALERT";
static char ble_last_payload[32] = "90.0,none";
static volatile float latest_angle_deg = DEFAULT_ANGLE_DEG;
static volatile bool angle_received = false;
static ble_gatt_chr_def ble_characteristics[2] = {};
static ble_gatt_svc_def ble_services[2] = {};

// -----------------------------------------------------------------------------
// Geekble Nano ESP32-S3 + INMP441 wiring that previously worked.
// -----------------------------------------------------------------------------
constexpr int I2S_PIN_WS  = 5;   // Geekble D4
constexpr int I2S_PIN_SCK = 6;   // Geekble D3
constexpr int I2S_PIN_SD  = 7;   // Geekble D2
constexpr float MIC_GAIN = 1.0f;

// -----------------------------------------------------------------------------
// v3.2 runtime RMS normalization: MUST match training model_info.json.
// -----------------------------------------------------------------------------
constexpr float RMS_GATE_PCM16 =50.0f;
constexpr float RMS_TARGET_PCM16 = 2500.0f;
constexpr float RMS_NORM_MIN_GAIN = 0.25f;
constexpr float RMS_NORM_MAX_GAIN = 8.0f;
constexpr float RMS_PEAK_LIMIT_FLOAT = 0.98f;
constexpr float MIN_CONFIDENCE = 0.75f;

// -----------------------------------------------------------------------------
// v3.2 model / feature parameters.
// librosa: mel spectrogram power=2, center=true, constant zero pad,
// Hann periodic window, Slaney mel norm, power_to_db(ref=1.0, top_db=80).
// -----------------------------------------------------------------------------
constexpr uint32_t SAMPLE_RATE = 16000;
constexpr uint32_t WINDOW_SECONDS = 2;
constexpr uint32_t SAMPLE_COUNT = SAMPLE_RATE * WINDOW_SECONDS;
constexpr uint16_t FFT_SIZE = 512;
constexpr uint16_t FFT_BINS = FFT_SIZE / 2 + 1;
constexpr uint16_t HOP_LENGTH = 256;
constexpr uint16_t CENTER_PADDING = FFT_SIZE / 2;
constexpr uint16_t MEL_BANDS = 64;
constexpr uint16_t TIME_FRAMES = 126;
constexpr uint32_t FEATURE_COUNT = static_cast<uint32_t>(MEL_BANDS) * TIME_FRAMES;
constexpr float MIN_DB = -80.0f;
constexpr float MAX_DB = 20.0f;
constexpr float POWER_FLOOR = 1.0e-10f;
constexpr float TOP_DB = 80.0f;

// Values exported in the uploaded v3.2 model_info.json.
constexpr float EXPECTED_INPUT_SCALE = 0.3710634410381317f;
constexpr int EXPECTED_INPUT_ZERO_POINT = 79;
constexpr float EXPECTED_OUTPUT_SCALE = 0.00390625f;
constexpr int EXPECTED_OUTPUT_ZERO_POINT = -128;

constexpr size_t TENSOR_ARENA_SIZE = 300000;
constexpr bool ENABLE_FEATURE_DEBUG = true;
constexpr bool ENABLE_OUTPUT_DEBUG = true;

uint8_t* tensor_arena = nullptr;
int16_t* audio_pcm = nullptr;
float* log_mel_buffer = nullptr;

float fft_real[FFT_SIZE];
float fft_imag[FFT_SIZE];
float power_spectrum[FFT_BINS];

i2s_chan_handle_t rx_handle = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;


// -----------------------------------------------------------------------------
// BLE GATT server
// -----------------------------------------------------------------------------
static void ble_start_advertising();

static int ble_gatt_access(uint16_t conn_handle, uint16_t attr_handle,
                           ble_gatt_access_ctxt* ctxt, void* arg) {
    (void)conn_handle;
    (void)arg;

    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR &&
        attr_handle == ble_char_val_handle) {
        const size_t len = std::strlen(ble_last_payload);
        const int rc = os_mbuf_append(ctxt->om, ble_last_payload, len);
        return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
    }

    return BLE_ATT_ERR_UNLIKELY;
}

static int ble_gap_event(ble_gap_event* event, void* arg) {
    (void)arg;

    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                ble_conn_handle = event->connect.conn_handle;
                ESP_LOGI(TAG, "BLE watch connected (handle=%u)",
                         static_cast<unsigned>(ble_conn_handle));
            } else {
                ESP_LOGW(TAG, "BLE connection failed (status=%d); advertising again",
                         event->connect.status);
                ble_start_advertising();
            }
            return 0;

        case BLE_GAP_EVENT_DISCONNECT:
            ESP_LOGI(TAG, "BLE watch disconnected (reason=%d)", event->disconnect.reason);
            ble_notify_enabled = false;
            ble_conn_handle = BLE_HS_CONN_HANDLE_NONE;
            ble_start_advertising();
            return 0;

        case BLE_GAP_EVENT_ADV_COMPLETE:
            ble_start_advertising();
            return 0;

        case BLE_GAP_EVENT_SUBSCRIBE:
            if (event->subscribe.attr_handle == ble_char_val_handle) {
                ble_notify_enabled = event->subscribe.cur_notify != 0;
                ble_conn_handle = event->subscribe.conn_handle;
                ESP_LOGI(TAG, "BLE notifications %s",
                         ble_notify_enabled ? "enabled" : "disabled");
            }
            return 0;

        case BLE_GAP_EVENT_MTU:
            ESP_LOGI(TAG, "BLE MTU=%u", static_cast<unsigned>(event->mtu.value));
            return 0;

        default:
            return 0;
    }
}

static void ble_start_advertising() {
    ble_hs_adv_fields adv_fields = {};
    adv_fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    adv_fields.uuids128 = &BLE_SERVICE_UUID;
    adv_fields.num_uuids128 = 1;
    adv_fields.uuids128_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&adv_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
        return;
    }

    // The device name is too long to fit alongside a 128-bit UUID in the
    // legacy advertising packet, so put the name in the scan response.
    ble_hs_adv_fields scan_rsp = {};
    scan_rsp.name = reinterpret_cast<uint8_t*>(ble_device_name);
    scan_rsp.name_len = std::strlen(ble_device_name);
    scan_rsp.name_is_complete = 1;

    rc = ble_gap_adv_rsp_set_fields(&scan_rsp);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_rsp_set_fields failed: %d", rc);
        return;
    }

    ble_gap_adv_params adv_params = {};
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

    rc = ble_gap_adv_start(ble_addr_type, nullptr, BLE_HS_FOREVER,
                           &adv_params, ble_gap_event, nullptr);
    if (rc != 0 && rc != BLE_HS_EALREADY) {
        ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
        return;
    }

    ESP_LOGI(TAG, "BLE advertising: %s", ble_device_name);
}

static void ble_on_sync() {
    const int rc = ble_hs_id_infer_auto(0, &ble_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed: %d", rc);
        return;
    }
    ble_start_advertising();
}

static void ble_on_reset(int reason) {
    ESP_LOGE(TAG, "NimBLE reset; reason=%d", reason);
}

static void ble_host_task(void* param) {
    (void)param;
    ESP_LOGI(TAG, "BLE Host Task started");
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static bool init_ble() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init failed: %s", esp_err_to_name(ret));
        return false;
    }

    uint8_t mac[6] = {};
    if (esp_read_mac(mac, ESP_MAC_WIFI_STA) == ESP_OK) {
        std::snprintf(ble_device_name, sizeof(ble_device_name),
                      "ESP32_AUDIO_ALERT_%02X%02X%02X",
                      mac[3], mac[4], mac[5]);
    }

    ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: %s", esp_err_to_name(ret));
        return false;
    }

    ble_svc_gap_init();
    ble_svc_gatt_init();

    // Characteristic: READ + NOTIFY, matching the Arduino sketch behavior.
    ble_characteristics[0].uuid = &BLE_CHARACTERISTIC_UUID.u;
    ble_characteristics[0].access_cb = ble_gatt_access;
    ble_characteristics[0].flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY;
    ble_characteristics[0].val_handle = &ble_char_val_handle;

    ble_services[0].type = BLE_GATT_SVC_TYPE_PRIMARY;
    ble_services[0].uuid = &BLE_SERVICE_UUID.u;
    ble_services[0].characteristics = ble_characteristics;

    int rc = ble_gatts_count_cfg(ble_services);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
        return false;
    }
    rc = ble_gatts_add_svcs(ble_services);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
        return false;
    }

    rc = ble_svc_gap_device_name_set(ble_device_name);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_svc_gap_device_name_set failed: %d", rc);
        return false;
    }

    ble_hs_cfg.reset_cb = ble_on_reset;
    ble_hs_cfg.sync_cb = ble_on_sync;

    nimble_port_freertos_init(ble_host_task);

    ESP_LOGI(TAG, "BLE initialized");
    ESP_LOGI(TAG, "Service UUID: %s", BLE_SERVICE_UUID_TEXT);
    ESP_LOGI(TAG, "Characteristic UUID: %s", BLE_CHARACTERISTIC_UUID_TEXT);
    return true;
}

static bool ble_send_sound(const char* sound_name) {
    if (!sound_name) return false;

    // Two values in one notification: angle,sound. The angle is the latest
    // value received from the other ESP32 over UART.
    const float angle = latest_angle_deg;
    if (!angle_received) {
        ESP_LOGW(TAG, "No UART angle received yet; using default %.1f ", angle);
    }
    std::snprintf(ble_last_payload, sizeof(ble_last_payload),
                  "%s, %.1f", sound_name, angle);

    if (ble_conn_handle == BLE_HS_CONN_HANDLE_NONE || !ble_notify_enabled) {
        ESP_LOGI(TAG, "BLE not subscribed; latest=%s", ble_last_payload);
        return false;
    }

    os_mbuf* om = ble_hs_mbuf_from_flat(
        ble_last_payload, std::strlen(ble_last_payload));
    if (!om) {
        ESP_LOGE(TAG, "BLE notify buffer allocation failed");
        return false;
    }

    const int rc = ble_gatts_notify_custom(
        ble_conn_handle, ble_char_val_handle, om);
    if (rc != 0) {
        ESP_LOGE(TAG, "BLE notify failed: rc=%d payload=%s", rc, ble_last_payload);
        return false;
    }

    ESP_LOGI(TAG, "BLE TX -> %s", ble_last_payload);
    return true;
}

// -----------------------------------------------------------------------------
// UART angle receiver
// Same behavior as tinyml_BLE.ino:
//   - UART1, 115200 bps, 8N1
//   - receive on Geekble RX (GPIO44), transmit on Geekble TX (GPIO43)
//   - one angle per line
//   - accept decimal values such as 135.0
//   - keep the latest valid angle for the next BLE notification
// -----------------------------------------------------------------------------
static bool parse_angle_line(char* line, float& angle_out) {
    if (!line) return false;

    // Arduino String.trim() equivalent.
    char* begin = line;
    while (*begin && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }

    char* finish = begin + std::strlen(begin);
    while (finish > begin &&
           std::isspace(static_cast<unsigned char>(*(finish - 1)))) {
        --finish;
    }
    *finish = '\0';

    if (*begin == '\0') return false;

    // Arduino String.toFloat() equivalent, but reject malformed garbage
    // instead of silently turning it into zero.
    char* parse_end = nullptr;
    const float value = std::strtof(begin, &parse_end);
    if (parse_end == begin) return false;

    while (*parse_end &&
           std::isspace(static_cast<unsigned char>(*parse_end))) {
        ++parse_end;
    }
    if (*parse_end != '\0') return false;

    if (!std::isfinite(value) ||
        value < ANGLE_MIN_DEG || value > ANGLE_MAX_DEG) {
        ESP_LOGW(TAG, "UART angle out of range: %.2f", value);
        return false;
    }

    angle_out = value;
    return true;
}

static void angle_uart_task(void* arg) {
    (void)arg;

    char line[32] = {};
    size_t used = 0;
    uint8_t ch = 0;

    while (true) {
        const int n = uart_read_bytes(
            ANGLE_UART_PORT, &ch, 1, pdMS_TO_TICKS(100));

        if (n <= 0) {
            continue;
        }

        // Same framing as Serial1.readStringUntil('\n').
        if (ch == '\n') {
            if (used == 0) {
                continue;
            }

            line[used] = '\0';

            float angle = 0.0f;
            if (parse_angle_line(line, angle)) {
                latest_angle_deg = angle;
                angle_received = true;
                ESP_LOGI(TAG, "UART angle RX -> %.1f deg", angle);
            } else {
                ESP_LOGW(TAG, "Invalid UART angle frame: '%s'", line);
            }

            used = 0;
            line[0] = '\0';
            continue;
        }

        // Ignore CR so both "\n" and "\r\n" senders work.
        if (ch == '\r') {
            continue;
        }

        if (used < sizeof(line) - 1) {
            line[used++] = static_cast<char>(ch);
        } else {
            ESP_LOGW(TAG, "UART angle frame too long; dropping");
            used = 0;
            line[0] = '\0';
        }
    }
}

static bool init_angle_uart() {
    uart_config_t cfg = {};
    cfg.baud_rate = ANGLE_UART_BAUD;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity = UART_PARITY_DISABLE;
    cfg.stop_bits = UART_STOP_BITS_1;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    cfg.source_clk = UART_SCLK_DEFAULT;

    esp_err_t err = uart_param_config(ANGLE_UART_PORT, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "uart_param_config failed: %s", esp_err_to_name(err));
        return false;
    }

    // Arduino equivalent:
    // Serial1.begin(115200, SERIAL_8N1, RX, TX);
    err = uart_set_pin(ANGLE_UART_PORT,
                       ANGLE_UART_TX_PIN,
                       ANGLE_UART_RX_PIN,
                       UART_PIN_NO_CHANGE,
                       UART_PIN_NO_CHANGE);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "uart_set_pin failed: %s", esp_err_to_name(err));
        return false;
    }

    // Small RX ring buffer is enough because each frame is only an angle + '\n'.
    err = uart_driver_install(ANGLE_UART_PORT, 512, 0, 0, nullptr, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "uart_driver_install failed: %s", esp_err_to_name(err));
        return false;
    }

    // Drop any stale bytes that arrived during boot.
    uart_flush_input(ANGLE_UART_PORT);

    const BaseType_t task_ok = xTaskCreate(
        angle_uart_task, "angle_uart", 3072, nullptr, 5, nullptr);
    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create angle UART task");
        return false;
    }

    ESP_LOGI(TAG,
             "Angle UART ready: UART%d %d bps 8N1, RX=GPIO%d(D0/RX) TX=GPIO%d(D1/TX), default=%.1f deg",
             static_cast<int>(ANGLE_UART_PORT), ANGLE_UART_BAUD,
             ANGLE_UART_RX_PIN, ANGLE_UART_TX_PIN, DEFAULT_ANGLE_DEG);
    return true;
}

struct RmsNormStats {
    float mean_pcm16 = 0.0f;
    float rms_before = 0.0f;
    float rms_after = 0.0f;
    float gain = 1.0f;
    float peak_after = 0.0f;
};

inline float clamp_float(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

inline int8_t clamp_int8(long value) {
    if (value < -128) return -128;
    if (value > 127) return 127;
    return static_cast<int8_t>(value);
}

size_t tensor_element_count(const TfLiteTensor* tensor) {
    if (!tensor || !tensor->dims) return 0;
    size_t count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        count *= static_cast<size_t>(tensor->dims->data[i]);
    }
    return count;
}

void print_tensor_shape(const char* name, const TfLiteTensor* tensor) {
    std::printf("%s shape: [", name);
    if (tensor && tensor->dims) {
        for (int i = 0; i < tensor->dims->size; ++i) {
            if (i) std::printf(", ");
            std::printf("%d", tensor->dims->data[i]);
        }
    }
    std::printf("]\n");
}

void* alloc_prefer_psram(size_t bytes, const char* name) {
    void* p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p) {
        std::printf("%s: %u bytes in PSRAM\n", name, static_cast<unsigned>(bytes));
        return p;
    }
    ESP_LOGW(TAG, "%s: PSRAM allocation failed; trying internal RAM", name);
    p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (p) {
        std::printf("%s: %u bytes in INTERNAL RAM\n", name, static_cast<unsigned>(bytes));
    }
    return p;
}

bool allocate_buffers() {
    // TFLM arena is fastest in internal SRAM. Fall back to PSRAM if the
    // contiguous internal block is not large enough. Audio/feature buffers
    // preferentially stay in PSRAM so I2S DMA/internal heap keeps headroom.
    tensor_arena = static_cast<uint8_t*>(
        heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (tensor_arena) {
        std::printf("Tensor Arena: %u bytes in INTERNAL RAM\n",
                    static_cast<unsigned>(TENSOR_ARENA_SIZE));
    } else {
        ESP_LOGW(TAG, "Internal Tensor Arena allocation failed; using PSRAM");
        tensor_arena = static_cast<uint8_t*>(
            heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (tensor_arena) {
            std::printf("Tensor Arena: %u bytes in PSRAM\n",
                        static_cast<unsigned>(TENSOR_ARENA_SIZE));
        }
    }

    audio_pcm = static_cast<int16_t*>(
        alloc_prefer_psram(SAMPLE_COUNT * sizeof(int16_t), "PCM buffer"));
    log_mel_buffer = static_cast<float*>(
        alloc_prefer_psram(FEATURE_COUNT * sizeof(float), "Log-Mel buffer"));

    if (!tensor_arena || !audio_pcm || !log_mel_buffer) {
        ESP_LOGE(TAG, "Buffer allocation failed. Check QSPI PSRAM settings.");
        return false;
    }
    std::memset(tensor_arena, 0, TENSOR_ARENA_SIZE);
    std::printf("Free PSRAM after buffers: %u bytes\n",
                static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
    std::printf("Free internal RAM       : %u bytes\n",
                static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
    return true;
}

void print_output_debug(const char* tag) {
    if (!ENABLE_OUTPUT_DEBUG || !output_tensor) return;
    std::printf("\n========== OUTPUT DEBUG ==========\n");
    std::printf("Tag: %s\n", tag);
    std::printf("Raw output INT8: ");
    for (int i = 0; i < SOUND_CLASS_COUNT; ++i) {
        std::printf("%d ", static_cast<int>(output_tensor->data.int8[i]));
    }
    std::printf("\nOutput scale: %.9f\n", output_tensor->params.scale);
    std::printf("Output zero point: %d\n", static_cast<int>(output_tensor->params.zero_point));
    std::printf("Dequantized output: ");
    for (int i = 0; i < SOUND_CLASS_COUNT; ++i) {
        const float value =
            (static_cast<int>(output_tensor->data.int8[i]) -
             static_cast<int>(output_tensor->params.zero_point)) * output_tensor->params.scale;
        std::printf("%.6f ", value);
    }
    std::printf("\n==================================\n");
}

bool init_model() {
    model = tflite::GetModel(sound_classifier_model);
    if (!model) {
        ESP_LOGE(TAG, "GetModel returned null");
        return false;
    }
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch: model=%lu runtime=%lu",
                 static_cast<unsigned long>(model->version()),
                 static_cast<unsigned long>(TFLITE_SCHEMA_VERSION));
        return false;
    }

    // esp-tflite-micro v1.3.5 does not ship all_ops_resolver.h.
    // Register exactly the 6 operators used by this v3.2 TFLite model:
    // SUB, CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX.
    static tflite::MicroMutableOpResolver<6> resolver;
    static bool resolver_initialized = false;
    if (!resolver_initialized) {
        if (resolver.AddSub() != kTfLiteOk ||
            resolver.AddConv2D() != kTfLiteOk ||
            resolver.AddMaxPool2D() != kTfLiteOk ||
            resolver.AddMean() != kTfLiteOk ||
            resolver.AddFullyConnected() != kTfLiteOk ||
            resolver.AddSoftmax() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to register TFLite Micro operators");
            return false;
        }
        resolver_initialized = true;
    }

    interpreter = new (std::nothrow) tflite::MicroInterpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    if (!interpreter) {
        ESP_LOGE(TAG, "MicroInterpreter allocation failed");
        return false;
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena=%u)",
                 static_cast<unsigned>(TENSOR_ARENA_SIZE));
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    if (!input_tensor || !output_tensor) {
        ESP_LOGE(TAG, "Input/output tensor missing");
        return false;
    }

    std::printf("TFLite model bytes: %u\n", sound_classifier_model_len);
    std::printf("Tensor Arena used: %u bytes\n",
                static_cast<unsigned>(interpreter->arena_used_bytes()));
    std::printf("Input quant : scale=%.9f zero=%d\n",
                input_tensor->params.scale, static_cast<int>(input_tensor->params.zero_point));
    std::printf("Output quant: scale=%.9f zero=%d\n",
                output_tensor->params.scale, static_cast<int>(output_tensor->params.zero_point));
    print_tensor_shape("Input", input_tensor);
    print_tensor_shape("Output", output_tensor);

    if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Model is not full INT8 input/output");
        return false;
    }
    if (tensor_element_count(input_tensor) != FEATURE_COUNT ||
        tensor_element_count(output_tensor) != SOUND_CLASS_COUNT) {
        ESP_LOGE(TAG, "Tensor size mismatch: expected input=%u output=%u",
                 static_cast<unsigned>(FEATURE_COUNT), static_cast<unsigned>(SOUND_CLASS_COUNT));
        return false;
    }

    const bool input_quant_ok =
        std::fabs(input_tensor->params.scale - EXPECTED_INPUT_SCALE) <= 0.00001f &&
        input_tensor->params.zero_point == EXPECTED_INPUT_ZERO_POINT;
    const bool output_quant_ok =
        std::fabs(output_tensor->params.scale - EXPECTED_OUTPUT_SCALE) <= 0.000001f &&
        output_tensor->params.zero_point == EXPECTED_OUTPUT_ZERO_POINT;

    if (!input_quant_ok || !output_quant_ok) {
        ESP_LOGE(TAG, "Quantization does not match uploaded v3.2 model_info.json");
        return false;
    }

    ESP_LOGI(TAG, "TFLite v3.2 model validation PASS");
    return true;
}

bool init_microphone() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 256;

    esp_err_t err = i2s_new_channel(&chan_cfg, nullptr, &rx_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(err));
        return false;
    }

    i2s_std_config_t std_cfg = {};
    std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
    std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    std_cfg.gpio_cfg.bclk = static_cast<gpio_num_t>(I2S_PIN_SCK);
    std_cfg.gpio_cfg.ws = static_cast<gpio_num_t>(I2S_PIN_WS);
    std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
    std_cfg.gpio_cfg.din = static_cast<gpio_num_t>(I2S_PIN_SD);
    std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.ws_inv = false;

    err = i2s_channel_init_std_mode(rx_handle, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode failed: %s", esp_err_to_name(err));
        return false;
    }
    err = i2s_channel_enable(rx_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_enable failed: %s", esp_err_to_name(err));
        return false;
    }

    // Flush startup samples.
    int32_t discard[256];
    for (int i = 0; i < 4; ++i) {
        size_t bytes_read = 0;
        err = i2s_channel_read(rx_handle, discard, sizeof(discard), &bytes_read,
                               pdMS_TO_TICKS(1000));
        if (err != ESP_OK || bytes_read == 0) {
            ESP_LOGE(TAG, "I2S startup read failed: %s bytes=%u",
                     esp_err_to_name(err), static_cast<unsigned>(bytes_read));
            ESP_LOGE(TAG, "Check INMP441: WS=GPIO%d SCK=GPIO%d SD=GPIO%d L/R=GND",
                 I2S_PIN_WS, I2S_PIN_SCK, I2S_PIN_SD);
            return false;
        }
    }

    ESP_LOGI(TAG, "INMP441 ready: 16 kHz, 32-bit I2S LEFT, raw>>16 to PCM16");
    return true;
}

bool capture_two_seconds(float& rms_out) {
    constexpr size_t CHUNK = 256;
    int32_t raw[CHUNK];
    uint32_t captured = 0;
    int16_t minimum = INT16_MAX;
    int16_t maximum = INT16_MIN;

    while (captured < SAMPLE_COUNT) {
        size_t bytes_read = 0;
        const esp_err_t err = i2s_channel_read(
            rx_handle, raw, sizeof(raw), &bytes_read, pdMS_TO_TICKS(1000));
        if (err != ESP_OK || bytes_read == 0) {
            ESP_LOGE(TAG, "I2S read failed: %s bytes=%u",
                     esp_err_to_name(err), static_cast<unsigned>(bytes_read));
            return false;
        }

        const size_t samples_read = bytes_read / sizeof(raw[0]);
        const size_t copy_count = std::min<size_t>(samples_read, SAMPLE_COUNT - captured);
        for (size_t i = 0; i < copy_count; ++i) {
            const int32_t pcm16_raw = raw[i] >> 16;
            float scaled = static_cast<float>(pcm16_raw) * MIC_GAIN;
            scaled = clamp_float(scaled, -32768.0f, 32767.0f);
            const int16_t sample = static_cast<int16_t>(std::lround(scaled));
            audio_pcm[captured++] = sample;
            minimum = std::min(minimum, sample);
            maximum = std::max(maximum, sample);
        }
    }

    if (minimum >= -1 && maximum <= 0) {
        ESP_LOGE(TAG, "Microphone data fixed at 0/-1; check wiring/soldering");
        return false;
    }

    int64_t sum = 0;
    for (uint32_t i = 0; i < SAMPLE_COUNT; ++i) sum += audio_pcm[i];
    const double mean = static_cast<double>(sum) / static_cast<double>(SAMPLE_COUNT);
    double squares = 0.0;
    for (uint32_t i = 0; i < SAMPLE_COUNT; ++i) {
        const double centered = static_cast<double>(audio_pcm[i]) - mean;
        squares += centered * centered;
    }
    rms_out = static_cast<float>(std::sqrt(squares / static_cast<double>(SAMPLE_COUNT)));
    std::printf("PCM16 min=%d max=%d raw_RMS=%.2f\n",
                static_cast<int>(minimum), static_cast<int>(maximum), rms_out);
    return true;
}

// Exact runtime equivalent of v3.2 normalize_rms_pcm16():
// DC removal -> target RMS -> gain clamp -> 0.98 peak limiter.
bool prepare_rms_normalization(float captured_rms, RmsNormStats& stats) {
    int64_t sum = 0;
    for (uint32_t i = 0; i < SAMPLE_COUNT; ++i) sum += audio_pcm[i];
    stats.mean_pcm16 = static_cast<float>(
        static_cast<double>(sum) / static_cast<double>(SAMPLE_COUNT));
    stats.rms_before = captured_rms;

    if (stats.rms_before < 1.0e-6f) {
        stats.gain = 1.0f;
        stats.rms_after = 0.0f;
        stats.peak_after = 0.0f;
        return false;
    }

    float gain = RMS_TARGET_PCM16 / stats.rms_before;
    gain = clamp_float(gain, RMS_NORM_MIN_GAIN, RMS_NORM_MAX_GAIN);

    float peak_before_float = 0.0f;
    for (uint32_t i = 0; i < SAMPLE_COUNT; ++i) {
        const float centered_float =
            (static_cast<float>(audio_pcm[i]) - stats.mean_pcm16) / 32768.0f;
        peak_before_float = std::max(peak_before_float, std::fabs(centered_float));
    }

    float peak_after = peak_before_float * gain;
    if (peak_after > RMS_PEAK_LIMIT_FLOAT) {
        const float limiter = RMS_PEAK_LIMIT_FLOAT / peak_after;
        gain *= limiter;
        peak_after *= limiter;
    }

    stats.gain = gain;
    stats.rms_after = stats.rms_before * gain;
    stats.peak_after = peak_after;

    std::printf("RMS normalize: %.2f -> %.2f (target %.0f), gain=%.4f, peak=%.4f\n",
                stats.rms_before, stats.rms_after, RMS_TARGET_PCM16,
                stats.gain, stats.peak_after);
    return true;
}

bool build_quantized_logmel_input(const RmsNormStats& norm) {
    if (!input_tensor || input_tensor->type != kTfLiteInt8) return false;

    float global_max_db = -1.0e30f;
    float raw_db_min = 1.0e30f;
    float raw_db_max = -1.0e30f;

    for (uint16_t frame = 0; frame < TIME_FRAMES; ++frame) {
        const int32_t frame_start = static_cast<int32_t>(frame) * HOP_LENGTH - CENTER_PADDING;

        for (uint16_t n = 0; n < FFT_SIZE; ++n) {
            const int32_t src = frame_start + n;
            float sample = 0.0f;  // librosa center=True, pad_mode='constant'
            if (src >= 0 && src < static_cast<int32_t>(SAMPLE_COUNT)) {
                sample =
                    (static_cast<float>(audio_pcm[src]) - norm.mean_pcm16) * norm.gain / 32768.0f;
                sample = clamp_float(sample, -1.0f, 1.0f);
            }
            fft_real[n] = sample * HANN_WINDOW[n];
            fft_imag[n] = 0.0f;
        }

        fft512_forward(fft_real, fft_imag);

        for (uint16_t bin = 0; bin < FFT_BINS; ++bin) {
            power_spectrum[bin] =
                fft_real[bin] * fft_real[bin] + fft_imag[bin] * fft_imag[bin];
        }

        for (uint16_t mel = 0; mel < MEL_BANDS; ++mel) {
            float mel_power = 0.0f;
            const uint16_t begin = MEL_FILTER_OFFSETS[mel];
            const uint16_t end = MEL_FILTER_OFFSETS[mel + 1];
            for (uint16_t item = begin; item < end; ++item) {
                mel_power += power_spectrum[MEL_FILTER_BINS[item]] * MEL_FILTER_WEIGHTS[item];
            }
            mel_power = std::max(mel_power, POWER_FLOOR);
            const float db = 10.0f * std::log10(mel_power);  // ref=1.0
            const uint32_t idx = static_cast<uint32_t>(mel) * TIME_FRAMES + frame;
            log_mel_buffer[idx] = db;
            raw_db_min = std::min(raw_db_min, db);
            raw_db_max = std::max(raw_db_max, db);
            global_max_db = std::max(global_max_db, db);
        }
    }

    const float top_db_floor = global_max_db - TOP_DB;
    const float scale = input_tensor->params.scale;
    const int zero = static_cast<int>(input_tensor->params.zero_point);
    if (scale <= 0.0f) return false;

    float clipped_min = 1.0e30f;
    float clipped_max = -1.0e30f;
    double clipped_sum = 0.0;
    int q_min = 127;
    int q_max = -128;

    for (uint32_t i = 0; i < FEATURE_COUNT; ++i) {
        float db = std::max(log_mel_buffer[i], top_db_floor);
        db = clamp_float(db, MIN_DB, MAX_DB);
        clipped_min = std::min(clipped_min, db);
        clipped_max = std::max(clipped_max, db);
        clipped_sum += db;

        const long q_long = std::lround(db / scale + static_cast<float>(zero));
        const int8_t q = clamp_int8(q_long);
        input_tensor->data.int8[i] = q;
        q_min = std::min(q_min, static_cast<int>(q));
        q_max = std::max(q_max, static_cast<int>(q));
    }

    if (ENABLE_FEATURE_DEBUG) {
        std::printf("\n========== FEATURE DEBUG ==========" "\n");
        std::printf("RMS before/after : %.2f / %.2f\n", norm.rms_before, norm.rms_after);
        std::printf("Normalization gain: %.5f\n", norm.gain);
        std::printf("Raw LogMel min/max: %.4f / %.4f\n", raw_db_min, raw_db_max);
        std::printf("globalMaxDb: %.4f, top_db floor: %.4f\n", global_max_db, top_db_floor);
        std::printf("Clipped LogMel min/max/mean: %.4f / %.4f / %.4f\n",
                    clipped_min, clipped_max,
                    static_cast<float>(clipped_sum / FEATURE_COUNT));
        std::printf("INT8 input min/max: %d / %d\n", q_min, q_max);
        std::printf("===================================\n");
    }
    return true;
}

bool run_live_inference(float scores[SOUND_CLASS_COUNT]) {
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "LIVE_AUDIO Invoke failed");
        return false;
    }
    print_output_debug("LIVE_AUDIO");

    for (int i = 0; i < SOUND_CLASS_COUNT; ++i) {
        scores[i] =
            (static_cast<int>(output_tensor->data.int8[i]) -
             static_cast<int>(output_tensor->params.zero_point)) * output_tensor->params.scale;
    }
    return true;
}

void print_prediction(const float scores[SOUND_CLASS_COUNT], const RmsNormStats& norm,
                      int64_t preprocess_us, int64_t inference_us) {
    int best = 0;
    for (int i = 1; i < SOUND_CLASS_COUNT; ++i) {
        if (scores[i] > scores[best]) best = i;
    }

    std::printf("\n========================================\n");
    std::printf("RMS raw -> normalized: %.2f -> %.2f (gain %.4f)\n",
                norm.rms_before, norm.rms_after, norm.gain);
    for (int i = 0; i < SOUND_CLASS_COUNT; ++i) {
        std::printf("%s: %.2f%%\n", SOUND_CLASS_NAMES[i], scores[i] * 100.0f);
    }
    std::printf("Preprocess: %.2f ms\n", preprocess_us / 1000.0);
    std::printf("Inference : %.2f ms\n", inference_us / 1000.0);
    if (scores[best] < MIN_CONFIDENCE) {
        std::printf("FINAL: uncertain (%.2f%%)\n", scores[best] * 100.0f);
    } else {
        std::printf("FINAL: %s (%.2f%%)\n", SOUND_CLASS_NAMES[best], scores[best] * 100.0f);
    }
    std::printf("========================================\n");
}

[[noreturn]] void stop_forever(const char* reason) {
    ESP_LOGE(TAG, "STOPPED: %s", reason);
    while (true) vTaskDelay(pdMS_TO_TICKS(2000));
}

}  // namespace

extern "C" void app_main(void) {
    std::printf("\n\n=======================================================\n");
    std::printf("Geekble ESP32-S3 Sound Classifier v3.2 RMS - ESP-IDF\n");
    std::printf("Model: horn / noise / siren | INT8 | Log-Mel 64x126x1\n");
    std::printf("Pipeline: RMS gate 150 -> normalize 2500 -> Log-Mel -> TFLite\n");
    std::printf("=======================================================\n");

    if (!allocate_buffers()) stop_forever("Memory allocation failed");
    if (!init_model()) stop_forever("TFLite v3.2 model initialization failed");
    if (!init_ble()) stop_forever("BLE initialization failed");
    if (!init_angle_uart()) stop_forever("Angle UART initialization failed");
    if (!init_microphone()) stop_forever("INMP441 initialization failed");

    ESP_LOGI(TAG, "Ready. Capturing 2-second windows...");

    while (true) {
        float raw_rms = 0.0f;
        if (!capture_two_seconds(raw_rms)) {
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        // Step 1: noise gate BEFORE normalization, exactly as runtime spec.
        if (raw_rms < RMS_GATE_PCM16) {
            std::printf("RMS %.2f < %.2f -> inference SKIPPED\n",
                        raw_rms, RMS_GATE_PCM16);
            continue;
        }

        // Step 2: DC removal + RMS target 2500 + gain limit + peak limiter.
        RmsNormStats norm;
        if (!prepare_rms_normalization(raw_rms, norm)) {
            std::printf("RMS normalization failed/empty input -> skipped\n");
            continue;
        }

        // Step 3: same librosa-compatible Log-Mel -> new model's INT8 quantization.
        const int64_t preprocess_start = esp_timer_get_time();
        if (!build_quantized_logmel_input(norm)) {
            ESP_LOGE(TAG, "Log-Mel preprocessing failed");
            continue;
        }
        const int64_t preprocess_end = esp_timer_get_time();

        // Step 4: v3.2 INT8 TFLite inference.
        float scores[SOUND_CLASS_COUNT] = {0.0f, 0.0f, 0.0f};
        const int64_t inference_start = esp_timer_get_time();
        if (!run_live_inference(scores)) continue;
        const int64_t inference_end = esp_timer_get_time();

        print_prediction(scores, norm,
                         preprocess_end - preprocess_start,
                         inference_end - inference_start);

        // Step 5: send danger sound result to the watch over BLE.
        // Noise and uncertain predictions are intentionally not notified.
        int best = 0;
        for (int i = 1; i < SOUND_CLASS_COUNT; ++i) {
            if (scores[i] > scores[best]) best = i;
        }
        if (scores[best] >= MIN_CONFIDENCE) {
            const char* sound = SOUND_CLASS_NAMES[best];
            if (std::strcmp(sound, "noise") != 0) {
                ble_send_sound(sound);
            }
        }
    }
}