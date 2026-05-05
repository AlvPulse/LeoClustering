import os
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import hashlib
import json

from src.config import load_config
from src.utils import set_seeds

def compute_snr(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    if len(audio) == 0:
        return 0.0

    num_frames = 1 + (len(audio) - frame_length) // hop_length
    if num_frames <= 0:
        energy = np.sum(audio**2)
        return float(10 * np.log10(energy + 1e-10))

    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(num_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0])
    )
    frame_energies = np.sum(frames**2, axis=1)

    peak_energy = np.max(frame_energies)
    median_energy = np.median(frame_energies)

    if median_energy == 0:
        return float(10 * np.log10(peak_energy + 1e-10))

    snr_db = 10 * np.log10((peak_energy + 1e-10) / (median_energy + 1e-10))
    return float(snr_db)

def generate_tone(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    noise = np.random.randn(len(tone)) * 0.1
    return tone + noise

def generate_noise(duration, sr=22050):
    return np.random.randn(int(sr * duration)) * 0.5

def main():
    config = load_config()
    set_seeds(config.seed)

    sr = config.dataset.sample_rate
    clip_dur = config.dataset.clip_duration_seconds

    raw_dir = Path(config.paths.raw_dir)
    processed_dir = Path(config.paths.processed_dir)
    results_dir = Path(config.paths.results_dir) / "step0" / "simulated"

    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["esc50", "dcase2024_t5"]

    for ds in datasets:
        manifest_data = []
        class_counts = {}
        total_duration = {}
        snr_estimates = {}

        ds_raw_dir = raw_dir / ds
        ds_raw_dir.mkdir(parents=True, exist_ok=True)
        ds_processed_dir = processed_dir / ds
        ds_processed_dir.mkdir(parents=True, exist_ok=True)

        # Increase to 100 clips so we have a meaningful split and multiple rare animal instances
        total_clips = 100

        for i in range(total_clips):
            # Rare class (2%), bird (49%), wind (49%)
            if i < 2:
                label = "rare_animal"
            elif i % 2 == 0:
                label = "bird"
            else:
                label = "wind"

            if label == "bird":
                audio = generate_tone(4000, clip_dur, sr)
            elif label == "wind":
                audio = generate_noise(clip_dur, sr)
            else:
                audio = generate_tone(1000, clip_dur, sr)

            file_name = f"sim_{i}.wav"
            file_path = ds_raw_dir / file_name
            sf.write(str(file_path), audio, sr)

            clip_id_str = f"{file_name}_0.0_{clip_dur}"
            clip_id = hashlib.sha256(clip_id_str.encode()).hexdigest()[:16]

            # Split logic: 70/20/10 roughly deterministic
            # To ensure representation, first of each goes to test, second to train, etc.
            # Using random choice per class
            rand_val = np.random.rand()
            if rand_val < 0.7:
                split = "train"
            elif rand_val < 0.9:
                split = "val"
            else:
                split = "test"

            manifest_data.append({
                "clip_id": clip_id,
                "dataset": ds,
                "source_file": f"{ds}/{file_name}",
                "start_time_seconds": 0.0,
                "duration_seconds": clip_dur,
                "sample_rate": sr,
                "label": label,
                "split": split
            })

            class_counts[label] = class_counts.get(label, 0) + 1
            total_duration[label] = total_duration.get(label, 0.0) + clip_dur

            snr = compute_snr(audio)
            snr_estimates.setdefault(label, []).append(snr)

        manifest_df = pd.DataFrame(manifest_data)

        # Ensure at least 1 rare animal in test and train manually if missed
        rare_mask = manifest_df['label'] == 'rare_animal'
        if not (manifest_df.loc[rare_mask, 'split'] == 'test').any():
            rare_indices = manifest_df[rare_mask].index
            if len(rare_indices) > 0: manifest_df.loc[rare_indices[0], 'split'] = 'test'
        if not (manifest_df.loc[rare_mask, 'split'] == 'train').any():
            rare_indices = manifest_df[rare_mask].index
            if len(rare_indices) > 1: manifest_df.loc[rare_indices[1], 'split'] = 'train'

        manifest_path = ds_processed_dir / "dataset_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Generated {manifest_path} with {len(manifest_df)} clips.")

        inventory = {
            "per_class_clip_counts": class_counts,
            "per_class_total_duration": total_duration,
            "sample_rate_distribution": {str(sr): total_clips},
            "clip_duration_distribution": {
                "min": clip_dur, "median": clip_dur, "max": clip_dur
            },
            "per_class_snr_estimate": {k: np.mean(v) for k, v in snr_estimates.items()}
        }

        with open(results_dir / f"{ds}_inventory.json", "w") as f:
            json.dump(inventory, f, indent=2)

if __name__ == "__main__":
    main()
