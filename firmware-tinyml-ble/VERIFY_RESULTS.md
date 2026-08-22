# Verification results

- Uploaded INT8 model size: 12,192 bytes
- Classes: horn / noise / siren
- Internal test CSV: 234 / 239 correct = 97.9079%
- Runtime RMS: gate 150, target 2500, gain 0.25..8.0, peak limiter 0.98
- Feature shape: 64 x 126 x 1
- TFLite input quantization: scale 0.3710634410381317, zero point 79
- TFLite output quantization: scale 0.00390625, zero point -128

## Independent preprocessing equivalence check

A deterministic 2-second PCM16 waveform was processed by:
1. Python/librosa 0.11.0 using the exact training settings, and
2. the generated C++ FFT512 + sparse Slaney mel implementation.

Result across all 8,064 Log-Mel values:
- max absolute difference: 0.0007476807 dB
- mean absolute difference: 0.0000316391 dB
- 99th percentile difference: 0.0004425049 dB
- every value differed by <= 0.001 dB

This validates the generated C++ Log-Mel preprocessing against the training pipeline.

## Environment limitation

The final ESP-IDF cross-build itself was not executed in the artifact environment because the ESP-IDF Xtensa toolchain is not installed there. Project structure, generated model bytes, preprocessing, configuration, and dependency version were checked.
