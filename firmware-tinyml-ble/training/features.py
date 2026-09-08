"""Log-Mel feature extraction that matches the on-device firmware exactly.

The firmware pipeline (main.cpp / model_info.json v3.2):
  1. 16 kHz mono PCM, 2 s window (32000 samples)
  2. DC removal, RMS normalize toward target 2500 pcm16, gain clamp, 0.98 peak limiter
  3. librosa-compatible mel spectrogram: n_fft=512, hop=256, n_mels=64,
     fmin=50, fmax=8000, power=2.0, center=True, constant zero pad,
     periodic Hann window, Slaney mel weights + Slaney norm
  4. power_to_db(ref=1.0, top_db=80)  ->  clip to [-80, 20] dB
  5. that float [64, 126] array is the model input; the model's own
     feature_normalization (Sub) layer centers it.

`verify_against_firmware()` checks this module's mel filterbank against the
generated C++ constants in main/mel_filterbank.h (VERIFY_RESULTS.md did the
same and reported <=0.001 dB agreement).
"""

from __future__ import annotations

import numpy as np
import librosa

from constants import (SAMPLE_RATE, WINDOW_SECONDS, SAMPLE_COUNT, N_FFT,
                       HOP_LENGTH, N_MELS, FMIN, FMAX, TIME_FRAMES, MIN_DB,
                       MAX_DB, TOP_DB, POWER_FLOOR, RMS_TARGET_PCM16,
                       RMS_NORM_MIN_GAIN, RMS_NORM_MAX_GAIN,
                       RMS_PEAK_LIMIT_FLOAT)

_MEL_BASIS = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    htk=False, norm="slaney",
).astype(np.float64)


def normalize_rms_pcm16(pcm16: np.ndarray) -> np.ndarray:
    """DC removal -> RMS target -> gain clamp -> 0.98 peak limiter.

    Returns a float32 waveform in roughly [-1, 1] (pcm16/32768 scale), the
    same thing build_quantized_logmel_input() feeds to the FFT on-device.
    """
    x = pcm16.astype(np.float64)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x * x))
    if rms < 1.0e-6:
        return np.zeros_like(x, dtype=np.float32)

    gain = RMS_TARGET_PCM16 / rms
    gain = float(np.clip(gain, RMS_NORM_MIN_GAIN, RMS_NORM_MAX_GAIN))

    centered = x / 32768.0
    peak_after = np.max(np.abs(centered)) * gain
    if peak_after > RMS_PEAK_LIMIT_FLOAT:
        gain *= RMS_PEAK_LIMIT_FLOAT / peak_after

    out = np.clip(centered * gain, -1.0, 1.0)
    return out.astype(np.float32)


def logmel_from_waveform(wave_pm1: np.ndarray) -> np.ndarray:
    """wave_pm1: float waveform already RMS-normalized to ~[-1, 1].

    Returns float32 [N_MELS, TIME_FRAMES] clipped log-mel in dB, exactly as
    the firmware produces it before INT8 quantization.
    """
    w = wave_pm1.astype(np.float64)
    if w.shape[0] < SAMPLE_COUNT:
        w = np.pad(w, (0, SAMPLE_COUNT - w.shape[0]))
    else:
        w = w[:SAMPLE_COUNT]

    stft = librosa.stft(
        w, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window="hann", center=True, pad_mode="constant",
    )
    power = (np.abs(stft) ** 2)[:, :TIME_FRAMES]
    mel_power = np.maximum(_MEL_BASIS @ power, POWER_FLOOR)

    db = 10.0 * np.log10(mel_power)                 # ref = 1.0
    db = np.maximum(db, db.max() - TOP_DB)          # top_db floor
    db = np.clip(db, MIN_DB, MAX_DB)
    return db.astype(np.float32)


def logmel_from_pcm16(pcm16: np.ndarray) -> np.ndarray:
    return logmel_from_waveform(normalize_rms_pcm16(pcm16))


def verify_against_firmware(header_path: str, tol_db: float = 0.02) -> None:
    """Rebuild the sparse mel filterbank from main/mel_filterbank.h and compare
    a random log-mel frame against this module. Raises AssertionError on drift.
    """
    import re

    text = open(header_path, "r", encoding="utf-8", errors="ignore").read()

    def _grab(name: str) -> list[float]:
        m = re.search(name + r"\[\d+\]\s*=\s*\{([^}]*)\}", text, re.S)
        body = m.group(1).replace("f", "").replace("\n", " ")
        return [float(v) for v in body.split(",") if v.strip()]

    offsets = [int(v) for v in _grab("MEL_FILTER_OFFSETS")]
    bins = [int(v) for v in _grab("MEL_FILTER_BINS")]
    weights = _grab("MEL_FILTER_WEIGHTS")

    fw_basis = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float64)
    for mel in range(N_MELS):
        for k in range(offsets[mel], offsets[mel + 1]):
            fw_basis[mel, bins[k]] = weights[k]

    max_row_diff = np.abs(fw_basis - _MEL_BASIS).max()
    assert max_row_diff < 1e-4, f"mel filterbank drift {max_row_diff:.2e}"

    rng = np.random.default_rng(0)
    wave = rng.standard_normal(SAMPLE_COUNT).astype(np.float32) * 0.05
    stft = librosa.stft(wave.astype(np.float64), n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=N_FFT, window="hann", center=True, pad_mode="constant")
    power = (np.abs(stft) ** 2)[:, :TIME_FRAMES]
    ours = 10.0 * np.log10(np.maximum(_MEL_BASIS @ power, POWER_FLOOR))
    theirs = 10.0 * np.log10(np.maximum(fw_basis @ power, POWER_FLOOR))
    d = np.abs(ours - theirs).max()
    assert d < tol_db, f"log-mel drift {d:.4f} dB exceeds {tol_db}"
    print(f"features.verify_against_firmware OK  (mel {max_row_diff:.2e}, logmel {d:.5f} dB)")


if __name__ == "__main__":
    import sys
    verify_against_firmware(sys.argv[1] if len(sys.argv) > 1
                            else "../main/mel_filterbank.h")
