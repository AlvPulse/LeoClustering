import json
import time
import uuid
import pandas as pd
from pathlib import Path

class LabelManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # We need a stable session_id per initialization, or reuse one.
        # For the demo, we generate a random session_id unless one exists.
        self.session_id = str(uuid.uuid4())[:8]
        self.log_path = self.output_dir / f"session_{self.session_id}.jsonl"
        self.state = {} # clip_id -> label_dict

        self._load_state()

    def _load_state(self):
        # Read all jsonl files in directory to replay full state across sessions
        for p in sorted(self.output_dir.glob("session_*.jsonl")):
            with open(p, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    # Replay logic: latest action overrides
                    self.state[data["clip_id"]] = data

    def append_action(self, user: str, action: str, clip_id: str, label: str, confidence: str, notes: str, tags: list):
        prev_label = self.state.get(clip_id, {}).get("label", None)

        entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "user": user,
            "action": action, # label, revise, uncertain
            "clip_id": clip_id,
            "label": label,
            "confidence": confidence,
            "notes": notes,
            "previous_label": prev_label,
            "tags": tags
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.state[clip_id] = entry

    def get_latest_labels(self):
        return self.state

    def export_parquet(self, out_path: Path):
        if not self.state:
            return False

        df = pd.DataFrame(list(self.state.values()))
        # keep only the latest per clip_id if we want a snapshot, but state dictionary already does this
        df.to_parquet(out_path, index=False)
        return True
