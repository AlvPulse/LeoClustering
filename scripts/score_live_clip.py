import argparse
import pickle
import soundfile as sf
import numpy as np
from pathlib import Path
from src.embeddings import get_extractor

def main():
    parser = argparse.ArgumentParser(description="Score a single audio clip using a trained few-shot Scorer.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--scorer", type=str, required=True, help="Path to the pickled Scorer object.")
    parser.add_argument("--representation", type=str, required=True, choices=["physics", "yamnet", "panns", "birdnet"])
    args = parser.parse_args()

    with open(args.scorer, "rb") as f:
        scorer = pickle.load(f)

    y, sr = sf.read(args.audio)
    duration = len(y) / sr

    if args.representation == "physics":
        from src.physics_features.extractor import extract_all_features
        from sklearn.preprocessing import StandardScaler
        features = extract_all_features(y, sr, duration)
        # We need to construct the feature array in the same column order as during training
        # But this is just a mockup for single-clip. The scorer expects a 2D array.
        dim_cols = [k for k in features.keys() if "dim_" in k or "spectral_" in k or "power" in k or "hz" in k or "time" in k or "rate" in k or "count" in k or "snr" in k or "fraction" in k or "rms" in k]
        vec = np.array([[features.get(c, 0.0) for c in dim_cols]])
        # Warning: For physics, the scorer expects scaled data. Since we don't have the scaler here,
        # we bypass scaling (in a real prod system, the scaler should be saved alongside the scorer).
        X = vec
    else:
        extractor = get_extractor(args.representation)
        vec = extractor.extract(y, sr)
        X = vec.reshape(1, -1)
        # Normalization
        from sklearn.preprocessing import normalize
        X = normalize(X, norm='l2', axis=1)

    score = scorer.score(X)
    print(f"Score for {args.audio} -> {score[0]}")

if __name__ == "__main__":
    main()
