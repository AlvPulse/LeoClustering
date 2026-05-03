import numpy as np
import pytest
from src.physics_features.quality import compute_quality_features
from src.physics_features.harmonic import compute_harmonic_features
from src.physics_features.bioacoustic import compute_bioacoustic_features

def test_snr_extraction():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)

    # Background noise
    noise = np.random.randn(len(t)) * 0.01

    # High energy pulse in the middle
    pulse = np.zeros_like(t)
    pulse[len(t)//2:len(t)//2 + 500] = 1.0

    signal = noise + pulse
    features = compute_quality_features(signal)

    snr = features["snr_db"]
    # We expect SNR to be strongly positive because the pulse is much louder than noise
    assert snr > 20.0

def test_hnr_extraction():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)

    # Pure tone (harmonic)
    tone = np.sin(400 * t * 2 * np.pi)

    # White noise (non-harmonic)
    noise = np.random.randn(len(t))

    # Highly harmonic signal
    features_tone = compute_harmonic_features(tone + noise * 0.01, sr)

    # Highly noisy signal
    features_noise = compute_harmonic_features(noise, sr)

    assert features_tone["harmonic_to_noise_ratio_db"] > features_noise["harmonic_to_noise_ratio_db"]

def test_syllable_count():
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)

    # Create 3 distinct syllables
    signal = np.zeros_like(t)

    # Pulse 1
    start1 = int(0.2 * sr)
    signal[start1:start1+1000] = np.random.randn(1000)

    # Pulse 2
    start2 = int(0.8 * sr)
    signal[start2:start2+1000] = np.random.randn(1000)

    # Pulse 3
    start3 = int(1.5 * sr)
    signal[start3:start3+1000] = np.random.randn(1000)

    features = compute_bioacoustic_features(signal, sr, duration)

    # We should detect approximately 3 syllables
    assert 2 <= features["syllable_count"] <= 4
