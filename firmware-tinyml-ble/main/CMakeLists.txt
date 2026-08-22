idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        nvs_flash
        bt
        esp_driver_i2s
        esp_driver_uart
        esp_timer
)