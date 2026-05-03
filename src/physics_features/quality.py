import numpy as np

def compute_quality_features(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Computes quality/SNR physics features from the audio signal.

    1. snr_db:
       - Physical meaning: Signal-to-noise ratio.
       - Computation: Ratio of peak-frame energy to median-frame energy, in dB.
       - Rationale: Helps filter out low-quality/inaudible recordings.

    2. background_noise_db:
       - Physical meaning: Background noise energy level.
       - Computation: Median-frame energy, in dB.
       - Rationale: Helps characterize the environmental noise conditions.
    """
    if len(y) == 0:
        return {
            "snr_db": np.nan,
            "background_noise_db": np.nan
        }

    num_frames = 1 + (len(y) - frame_length) // hop_length
    if num_frames <= 0:
        energy = np.sum(y**2)
        bg_db = 10 * np.log10(energy + 1e-10)
        return {
            "snr_db": 0.0,
            "background_noise_db": float(bg_db)
        }

    frames = np.lib.stride_tricks.as_strided(
        y, shape=(num_frames, frame_length),
        strides=(y.strides[0] * hop_length, y.strides[0])
    )
    frame_energies = np.sum(frames**2, axis=1)

    peak_energy = np.max(frame_energies)
    median_energy = np.median(frame_energies)

    background_noise_db = 10 * np.log10(median_energy + 1e-10)

    if median_energy == 0:
        snr_db = 10 * np.log10(peak_energy + 1e-10)
    else:
        snr_db = 10 * np.log10((peak_energy + 1e-10) / (median_energy + 1e-10))

    return {
        "snr_db": float(snr_db),
        "background_noise_db": float(background_noise_db)
    }
