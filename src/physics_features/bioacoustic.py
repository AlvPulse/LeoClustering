import numpy as np
import librosa

def compute_bioacoustic_features(y: np.ndarray, sr: int, duration: float) -> dict:
    """
    Computes bioacoustic-specific structure features from the audio signal.

    1. syllable_count:
       - Physical meaning: Distinct energy bursts in the audio.
       - Computation: Peak picking on onset envelope.
       - Rationale: Captures the temporal structure of animal calls (e.g., trills, chirps).

    2. syllable_rate_hz:
       - Physical meaning: Syllables per second.
       - Computation: syllable_count / clip_duration_seconds.
       - Rationale: Helps distinguish fast-paced vocalizations.

    3. mean_syllable_duration_seconds:
       - Physical meaning: Average length of a syllable.
       - Computation: Approximated using the temporal spacing of peaks.
       - Rationale: Distinguishes short clicks from long drawn-out calls.

    4. frequency_modulation_slope_hz_per_s:
       - Physical meaning: Linear fit slope of dominant frequency over time.
       - Computation: Polynomial fit (degree 1) on the argmax of the magnitude spectrum across time frames.
       - Rationale: Distinguishes up-sweeps, down-sweeps, and flat tones.
    """
    if len(y) == 0:
        return {
            "syllable_count": np.nan,
            "syllable_rate_hz": np.nan,
            "mean_syllable_duration_seconds": np.nan,
            "frequency_modulation_slope_hz_per_s": np.nan
        }

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)

    syllable_count = len(peaks)
    syllable_rate_hz = syllable_count / duration if duration > 0 else 0.0

    mean_syllable_duration_seconds = 0.0
    if syllable_count > 0:
        mean_syllable_duration_seconds = min(0.1, duration / syllable_count)

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    dom_freq_idx_per_frame = np.argmax(S, axis=0)
    dom_freqs = freqs[dom_freq_idx_per_frame]
    times = librosa.frames_to_time(np.arange(len(dom_freqs)), sr=sr)

    if len(times) > 1:
        slope, _ = np.polyfit(times, dom_freqs, 1)
        fm_slope = slope
    else:
        fm_slope = np.nan

    return {
        "syllable_count": float(syllable_count),
        "syllable_rate_hz": float(syllable_rate_hz),
        "mean_syllable_duration_seconds": float(mean_syllable_duration_seconds),
        "frequency_modulation_slope_hz_per_s": float(fm_slope) if not np.isnan(fm_slope) else np.nan
    }
