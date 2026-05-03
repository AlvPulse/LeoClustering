import numpy as np
import librosa

def compute_harmonic_features(y: np.ndarray, sr: int) -> dict:
    """
    Computes harmonic physics features from the audio signal.

    1. harmonic_to_noise_ratio_db:
       - Physical meaning: Ratio of harmonic energy to noise energy.
       - Computation: librosa.effects.hpss separation to extract harmonic and percussive parts.
       - Rationale: Measures the degree of acoustic periodicity (voicedness).

    2. pitch_mean_hz:
       - Physical meaning: Fundamental frequency of the signal.
       - Computation: Mean of voiced frames from librosa.pyin pitch tracker.
       - Rationale: Key for characterizing the melody/structure of vocalizations.

    3. pitch_std_hz:
       - Physical meaning: Variation in the fundamental frequency.
       - Computation: Standard deviation of voiced frames from librosa.pyin.
       - Rationale: Represents frequency modulation and variability in a call.

    4. pitch_voiced_fraction:
       - Physical meaning: Fraction of frames classified as voiced.
       - Computation: Count of voiced frames divided by total frames from librosa.pyin.
       - Rationale: Determines how much of the signal contains a strong fundamental pitch.
    """
    if len(y) == 0:
        return {
            "harmonic_to_noise_ratio_db": np.nan,
            "pitch_mean_hz": np.nan,
            "pitch_std_hz": np.nan,
            "pitch_voiced_fraction": np.nan
        }

    y_harm, y_perc = librosa.effects.hpss(y)
    energy_harm = np.sum(y_harm**2)
    energy_noise = np.sum(y_perc**2)
    if energy_noise > 0:
        hnr = 10 * np.log10(energy_harm / energy_noise + 1e-10)
    else:
        hnr = np.nan

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    if f0 is not None and np.any(voiced_flag):
        voiced_f0 = f0[voiced_flag]
        pitch_mean_hz = np.mean(voiced_f0)
        pitch_std_hz = np.std(voiced_f0)
        pitch_voiced_fraction = np.sum(voiced_flag) / len(voiced_flag)
    else:
        pitch_mean_hz = np.nan
        pitch_std_hz = np.nan
        pitch_voiced_fraction = 0.0

    return {
        "harmonic_to_noise_ratio_db": float(hnr) if not np.isnan(hnr) else np.nan,
        "pitch_mean_hz": float(pitch_mean_hz) if not np.isnan(pitch_mean_hz) else np.nan,
        "pitch_std_hz": float(pitch_std_hz) if not np.isnan(pitch_std_hz) else np.nan,
        "pitch_voiced_fraction": float(pitch_voiced_fraction)
    }
