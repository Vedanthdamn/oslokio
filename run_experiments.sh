#!/usr/bin/env bash
# Full experiment pipeline. Assumes a population already exists (see README for
# generation). Everything downstream of generation is cheap enough to re-run.
set -euo pipefail

PY=${PY:-.venv/bin/python}
MODELS=${MODELS:-data/models_v2}
OUT=${OUT:-data/results}
SEEDS=${SEEDS:-10}

mkdir -p "$OUT"

echo "==> extracting features"
$PY -m src.build_feature_matrix --models-dir "$MODELS" --out "$OUT/features.csv"
$PY -m src.build_feature_matrix --models-dir "$MODELS" --out "$OUT/features_depthinv.csv" \
    --extractor depth_invariant
OSLOKIO_NO_WEIGHT_NORM=1 $PY -m src.build_feature_matrix --models-dir "$MODELS" \
    --out "$OUT/features_rawscale.csv"

echo "==> generalization across every held-out axis"
$PY -m src.run_generalization --features-csv "$OUT/features.csv" \
    --out "$OUT/generalization.json" --n-seeds "$SEEDS"

echo "==> ablation: depth-invariant features"
$PY -m src.run_generalization --features-csv "$OUT/features_depthinv.csv" \
    --out "$OUT/generalization_depthinv.json" --n-seeds "$SEEDS"

echo "==> ablation: no per-layer weight normalization"
$PY -m src.run_generalization --features-csv "$OUT/features_rawscale.csv" \
    --out "$OUT/generalization_rawscale.json" --n-seeds "$SEEDS"

echo "==> cross-family attack transfer"
$PY -m src.cross_family --features-csv "$OUT/features.csv" \
    --out "$OUT/cross_family.json" --n-seeds "$SEEDS"

echo "==> few-shot threshold calibration on an unseen architecture"
$PY -m src.calibrate_fewshot --features-csv "$OUT/features.csv" --depth 4 \
    --out "$OUT/fewshot_calibration.json" --n-seeds 20

echo "==> which weight statistics carry the signal"
$PY -m src.analyze_features --features-csv "$OUT/features.csv" \
    --out "$OUT/feature_analysis.json" --n-seeds 5

echo "==> done; results in $OUT"
