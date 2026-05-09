import numpy as np
import pandas as pd
from src.demo.data import compute_structured_samples

def test_structured_samples_gaussian():
    # Make a synthetic cluster in a way that respects cosine distance logic
    np.random.seed(42)
    # Cosine distance operates on angles. Let's make cluster 0 a tight bundle of vectors
    # near [1, 0, 0, ...]
    c0 = np.random.randn(50, 10) * 0.05
    c0[:, 0] += 1.0

    # Intentionally create an outlier point that is still technically in cluster 0
    # but at a slightly different angle
    c0[0] = np.zeros(10)
    c0[0][0] = 1.0
    c0[0][1] = 0.5  # pushes it away angularly

    # Cluster 1 (other), pointing near [0, 1, 0, ...]
    c1 = np.random.randn(50, 10) * 0.05
    c1[:, 1] += 1.0

    X = np.vstack([c0, c1])

    emb_df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(10)])
    emb_df["clip_id"] = [f"clip_{i}" for i in range(100)]

    assign_df = pd.DataFrame({
        "clip_id": [f"clip_{i}" for i in range(100)],
        "cluster_id": [0]*50 + [1]*50
    })

    samples = compute_structured_samples(emb_df, assign_df)

    c0_samples = samples[0]

    # Medoid shouldn't be the outlier
    assert c0_samples["medoid"] != "clip_0"

    # Outliers should contain the extreme point (clip_0)
    assert "clip_0" in c0_samples["outliers"]
