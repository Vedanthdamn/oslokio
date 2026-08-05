import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.features import extract_features_from_file
from src.features_depth import extract_features_depth_invariant_from_file

EXTRACTORS = {
    "absolute": extract_features_from_file,
    "depth_invariant": extract_features_depth_invariant_from_file,
}


def build_feature_matrix(models_dir: Path, extractor: str = "absolute") -> pd.DataFrame:
    extract = EXTRACTORS[extractor]
    rows = []
    # a directory mid-write has weights but no metadata yet; skipping those makes this
    # safe to run against a population that is still being generated.
    model_dirs = sorted(
        p for p in models_dir.iterdir()
        if p.is_dir() and (p / "metadata.json").exists() and (p / "weights.pt").exists()
    )

    for model_dir in tqdm(model_dirs, desc=f"extracting features ({extractor})"):
        metadata = json.load(open(model_dir / "metadata.json"))
        features = extract(str(model_dir / "weights.pt"))
        row = {
            "model_id": model_dir.name,
            "label": metadata["label"],
            "clean_test_acc": metadata["clean_test_acc"],
            "backdoor_success_rate": metadata["backdoor_success_rate"],
            "n_conv_layers": len(metadata["architecture"]["channels"]),
        }
        if metadata["trigger"] is not None:
            for k, v in metadata["trigger"].items():
                row[f"trigger_{k}"] = v
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--out", type=str, default="data/features.csv")
    parser.add_argument("--extractor", choices=sorted(EXTRACTORS), default="absolute")
    args = parser.parse_args()

    df = build_feature_matrix(Path(args.models_dir), args.extractor)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows, {len(df.columns)} columns to {args.out}")
