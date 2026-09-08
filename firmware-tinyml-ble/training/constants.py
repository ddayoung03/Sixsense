"""Plain feature/quant constants shared across the pipeline.

Kept import-free (no numpy/librosa/tf) so modules that must not load librosa in
the same process as TensorFlow -- importing both triggers an OpenMP double-free
in this WSL env -- can still get the shapes.
"""

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
SAMPLE_COUNT = int(SAMPLE_RATE * WINDOW_SECONDS)  # 32000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
FMIN = 50.0
FMAX = 8000.0
TIME_FRAMES = 126
MIN_DB = -80.0
MAX_DB = 20.0
TOP_DB = 80.0
POWER_FLOOR = 1.0e-10

RMS_TARGET_PCM16 = 2500.0
RMS_NORM_MIN_GAIN = 0.25
RMS_NORM_MAX_GAIN = 8.0
RMS_PEAK_LIMIT_FLOAT = 0.98
