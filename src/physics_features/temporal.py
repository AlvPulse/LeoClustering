import numpy as np
import librosa

def compute_temporal_features(y: np.ndarray, sr: int, duration: float) -> dict:
    """
    Computes temporal physics features from the audio signal.

    1. clip_duration_seconds:
       - Physical meaning: The total duration of the clip.
       - Computation: Retrieved from manifest.
       - Rationale: Contextual length of the vocalization event.

    2. rms_energy_db:
       - Physical meaning: Root mean square of the audio signal converted to dB.
       - Computation: 10 * log10(mean(RMS)^2) using librosa.feature.rms.
       - Rationale: General measure of signal loudness and intensity.

    3. attack_time_seconds:
       - Physical meaning: Time to reach peak envelope from start.
       - Computation: argmax of librosa onset strength converted to time.
       - Rationale: Helps distinguish percussive/impulsive sounds from sustained ones.

    4. decay_time_seconds:
       - Physical meaning: Time from peak envelope to the end of the clip.
       - Computation: duration - attack_time_seconds.
       - Rationale: Measures the fade-out or reverberation characteristics.

    5. zero_crossing_rate:
       - Physical meaning: Rate at which the signal changes sign.
       - Computation: mean of librosa.feature.zero_crossing_rate.
       - Rationale: Correlates strongly with the noise component of the signal.
    """
    if len(y) == 0:
        return {
            "clip_duration_seconds": duration,
            "rms_energy_db": np.nan,
            "attack_time_seconds": np.nan,
            "decay_time_seconds": np.nan,
            "zero_crossing_rate": np.nan
        }

    rms = librosa.feature.rms(y=y)[0]
    mean_rms = np.mean(rms)
    rms_energy_db = 10 * np.log10(mean_rms**2 + 1e-10)

    envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    if len(envelope) > 0:
        peak_frame = np.argmax(envelope)
        frames_to_time = librosa.frames_to_time(np.arange(len(envelope)), sr=sr)
        attack_time_seconds = frames_to_time[peak_frame]
        decay_time_seconds = duration - attack_time_seconds
    else:
        attack_time_seconds = np.nan
        decay_time_seconds = np.nan

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zero_crossing_rate = np.mean(zcr)

    return {
        "clip_duration_seconds": float(duration),
        "rms_energy_db": float(rms_energy_db),
        "attack_time_seconds": float(attack_time_seconds),
        "decay_time_seconds": float(decay_time_seconds),
        "zero_crossing_rate": float(zero_crossing_rate)
    }
