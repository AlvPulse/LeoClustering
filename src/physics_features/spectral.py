import numpy as np
import librosa
import scipy.stats

def compute_spectral_features(y: np.ndarray, sr: int) -> dict:
    """
    Computes spectral physics features from the audio signal.

    1. spectral_centroid_hz:
       - Physical meaning: Frequency center of mass of the spectrum.
       - Computation: librosa.feature.spectral_centroid
       - Rationale: Higher for noisy/bright sounds, lower for tonal/dull sounds.

    2. spectral_flatness:
       - Physical meaning: Ratio of geometric mean to arithmetic mean of the power spectrum.
       - Computation: librosa.feature.spectral_flatness
       - Rationale: Differentiates tonal (close to 0) from noise-like (close to 1) signals.

    3. dominant_frequency_hz:
       - Physical meaning: Peak of the average spectrum.
       - Computation: argmax of mean magnitude spectrum
       - Rationale: Identifies the primary frequency of communication for an animal.

    4. low_band_power_db (0-500 Hz):
       - Physical meaning: Energy within the low frequency band.
       - Computation: Sum of squared mean spectrum magnitudes.
       - Rationale: Captures low-frequency rumbles or noise.

    5. mid_band_power_db (500-2000 Hz):
       - Physical meaning: Energy within the mid frequency band.
       - Computation: Sum of squared mean spectrum magnitudes.
       - Rationale: Captures mid-range animal vocalizations.

    6. high_band_power_db (2000-8000 Hz):
       - Physical meaning: Energy within the high frequency band.
       - Computation: Sum of squared mean spectrum magnitudes.
       - Rationale: Captures high-pitched chirps and insect noise.

    7. low_high_power_ratio_db:
       - Physical meaning: Ratio of low to high frequency band power.
       - Computation: low_band_power_db - high_band_power_db
       - Rationale: Measures the relative balance between low and high frequencies.

    8. spectral_entropy:
       - Physical meaning: Shannon entropy of the normalized magnitude spectrum.
       - Computation: scipy.stats.entropy of normalized spectrum.
       - Rationale: Measures the peakiness or flatness of the spectrum.

    9. bandwidth_hz:
       - Physical meaning: Frequency range containing 90% of spectral energy.
       - Computation: Difference between the 95th and 5th percentiles of cumulative spectral energy.
       - Rationale: Differentiates narrowband whistles from broadband clicks/noise.
    """
    if len(y) == 0:
        return {
            "spectral_centroid_hz": np.nan,
            "spectral_flatness": np.nan,
            "dominant_frequency_hz": np.nan,
            "low_band_power_db": np.nan,
            "mid_band_power_db": np.nan,
            "high_band_power_db": np.nan,
            "low_high_power_ratio_db": np.nan,
            "spectral_entropy": np.nan,
            "bandwidth_hz": np.nan
        }

    S = np.abs(librosa.stft(y))
    S_mean = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spectral_centroid_hz = np.nanmean(centroid)

    flatness = librosa.feature.spectral_flatness(S=S)
    spectral_flatness = np.nanmean(flatness)

    dom_idx = np.argmax(S_mean)
    dominant_frequency_hz = freqs[dom_idx]

    def band_power(S_mean, freqs, low, high):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return -100.0
        energy = np.sum(S_mean[mask] ** 2)
        return 10 * np.log10(energy + 1e-10)

    low_band_power_db = band_power(S_mean, freqs, 0, 500)
    mid_band_power_db = band_power(S_mean, freqs, 500, 2000)
    high_band_power_db = band_power(S_mean, freqs, 2000, 8000)
    low_high_power_ratio_db = low_band_power_db - high_band_power_db

    p = S_mean / (np.sum(S_mean) + 1e-10)
    spectral_entropy = scipy.stats.entropy(p)

    energy = S_mean ** 2
    cumsum = np.cumsum(energy)
    total_energy = cumsum[-1]
    if total_energy == 0:
        bandwidth_hz = np.nan
    else:
        lower_idx = np.searchsorted(cumsum, 0.05 * total_energy)
        upper_idx = np.searchsorted(cumsum, 0.95 * total_energy)
        bandwidth_hz = freqs[upper_idx] - freqs[lower_idx]

    return {
        "spectral_centroid_hz": float(spectral_centroid_hz),
        "spectral_flatness": float(spectral_flatness),
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "low_band_power_db": float(low_band_power_db),
        "mid_band_power_db": float(mid_band_power_db),
        "high_band_power_db": float(high_band_power_db),
        "low_high_power_ratio_db": float(low_high_power_ratio_db),
        "spectral_entropy": float(spectral_entropy),
        "bandwidth_hz": float(bandwidth_hz)
    }
