#!/usr/bin/env python3
"""
Export LogisticRegression weights as raw float32 binary files for use by the
TypeScript inference engine (avoids onnxruntime-node native-binary issues on
Vercel Lambda).

Reads:   ../public/model/detector.pkl
Writes:  ../public/model/coef.bin       — float32[n_classes × n_features], row-major
         ../public/model/intercept.bin  — float32[n_classes]
"""

import pickle
import struct
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent.parent / "public" / "model"
PKL_PATH  = MODEL_DIR / "detector.pkl"

if not PKL_PATH.exists():
    print(f"ERROR: {PKL_PATH} not found. Run train.py first.", file=sys.stderr)
    sys.exit(1)

with open(PKL_PATH, "rb") as f:
    pipeline = pickle.load(f)

lr     = pipeline.named_steps["clf"]
coef   = lr.coef_.astype(np.float32)      # (n_classes, n_features)
intercept = lr.intercept_.astype(np.float32)   # (n_classes,)

coef_path = MODEL_DIR / "coef.bin"
intercept_path = MODEL_DIR / "intercept.bin"

coef.tofile(coef_path)
intercept.tofile(intercept_path)

print(f"coef      {coef.shape}  {coef.nbytes / 1024:.0f} KB  -> {coef_path}")
print(f"intercept {intercept.shape}  {intercept.nbytes} B    -> {intercept_path}")
print("Done. TypeScript can now run LogReg inference without onnxruntime.")
