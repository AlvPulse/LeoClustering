import numpy as np
import librosa
from src.physics_features.spectral import compute_spectral_features
from src.physics_features.temporal import compute_temporal_features
from src.physics_features.harmonic import compute_harmonic_features
from src.physics_features.quality import compute_quality_features
from src.physics_features.bioacoustic import compute_bioacoustic_features

def extract_all_features(y: np.ndarray, sr: int, duration: float) -> dict:
    features = {}
    features.update(compute_spectral_features(y, sr))
    features.update(compute_temporal_features(y, sr, duration))
    features.update(compute_harmonic_features(y, sr))
    features.update(compute_quality_features(y))
    features.update(compute_bioacoustic_features(y, sr, duration))
    return features
