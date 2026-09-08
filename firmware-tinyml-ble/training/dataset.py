"""Build 2-second training windows.

The core fix for weak horn detection: a real car horn heard on the street is a
short (~0.3-1.5 s) tonal burst buried in continuous traffic / wind / engine
noise, often at low SNR. The original v3.2 data was mostly isolated, cleanly
recorded horn clips with only gentle added noise (SNR 18-35 dB), so the model
learned "horn = clear tone over near-silence" and misses real-world horns.

  horn / siren : random-offset event  +  real background bed  at a realistic
                 (often low) SNR, event frequently occupying only part of the
                 2 s window.
  noise        : the background material itself as a plain 2 s window.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from constants import SAMPLE_RATE, SAMPLE_COUNT

CLASSES = ["horn", "noise", "siren"]


@dataclasses.dataclass
class ClassConfig:
    bg_prob: float                       # fraction of events given a background bed
    snr_db: tuple[float, float]          # event-over-background SNR when a bed is mixed
    keep_frac: tuple[float, float]       # event length kept, fraction of original
    gain_db: tuple[float, float] = (-6.0, 6.0)   # whole-window gain jitter


# Balance: hard enough that the model must learn the horn's spectral signature
# through street noise, but not so buried that it collapses to predicting
# "noise". The original v3.2 used SNR 18-35 dB (background barely audible) and
# no length variation - that is what this widens.
DEFAULT_CONFIG = {
    "horn": ClassConfig(bg_prob=0.8, snr_db=(0.0, 18.0), keep_frac=(0.4, 1.0)),
    "siren": ClassConfig(bg_prob=0.6, snr_db=(3.0, 20.0), keep_frac=(0.5, 1.0)),
}


def load_mono_16k(path: str | Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    return np.ascontiguousarray(wav, dtype=np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def _fit_length(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Exactly SAMPLE_COUNT samples: tile if short, random-crop if long."""
    if x.shape[0] < SAMPLE_COUNT:
        x = np.tile(x, int(np.ceil(SAMPLE_COUNT / max(x.shape[0], 1))))
    start = rng.randint(0, x.shape[0] - SAMPLE_COUNT) if x.shape[0] > SAMPLE_COUNT else 0
    return x[start:start + SAMPLE_COUNT].copy()


def _finish(mix: np.ndarray, gain_db: tuple[float, float],
            rng: random.Random, to_pcm16: bool) -> np.ndarray:
    mix = mix * (10.0 ** (rng.uniform(*gain_db) / 20.0))
    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        mix = mix / peak
    if not to_pcm16:
        return mix.astype(np.float32)
    return np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)


def plain_window(clip: np.ndarray, rng: random.Random,
                 gain_db: tuple[float, float] = (-6.0, 6.0),
                 to_pcm16: bool = True) -> np.ndarray:
    """A 2 s window straight from `clip` (used for the noise class)."""
    return _finish(_fit_length(clip.astype(np.float32), rng), gain_db, rng, to_pcm16)


def event_window(event: np.ndarray, cfg: ClassConfig, rng: random.Random,
                 backgrounds: list[np.ndarray] | None,
                 to_pcm16: bool = True) -> np.ndarray:
    """A 2 s window: trimmed event at a random offset + optional background bed."""
    ev = event.astype(np.float32)

    keep = rng.uniform(*cfg.keep_frac)
    if 0.0 < keep < 1.0 and ev.shape[0] > 400:
        n = max(int(ev.shape[0] * keep), 400)
        s = rng.randint(0, ev.shape[0] - n)
        ev = ev[s:s + n]
    ev = ev[:SAMPLE_COUNT]

    canvas = np.zeros(SAMPLE_COUNT, dtype=np.float32)
    off = rng.randint(0, SAMPLE_COUNT - ev.shape[0]) if ev.shape[0] < SAMPLE_COUNT else 0
    canvas[off:off + ev.shape[0]] = ev

    mix = canvas
    if backgrounds and rng.random() < cfg.bg_prob:
        bed = _fit_length(backgrounds[rng.randrange(len(backgrounds))], rng)
        snr = rng.uniform(*cfg.snr_db)
        bed = bed * (_rms(canvas) / (10.0 ** (snr / 20.0)) / _rms(bed))
        mix = canvas + bed

    return _finish(mix, cfg.gain_db, rng, to_pcm16)


def iter_wavs(root: str | Path) -> list[Path]:
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)
