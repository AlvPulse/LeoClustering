import sys
import pandas as pd
import json
import hashlib
from pathlib import Path

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]

res_df = pd.read_csv("results/step3/20260505_100212/esc50/clustering_results.csv")
for _, row in res_df.iterrows():
    params = json.loads(row['hyperparameters'])
    h = stable_hash(params)
    print(h)
    break
