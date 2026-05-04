"""
SVM-RDKit Descriptor Predictor
Predict molecular activity from SMILES using a pre-trained SVM model.

Usage:
    python predict.py --input smile.csv --output result.csv
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def _load_descriptor_config(path: str):
    """Load the fixed descriptor name list and build a name -> func map."""
    with open(path, "r") as f:
        fixed_names = json.load(f)
    func_map = dict(Descriptors.descList)
    return fixed_names, func_map


def _compute_descriptors(mol, fixed_names, func_map):
    """Compute RDKit descriptors for a single molecule in fixed order."""
    features = []
    for name in fixed_names:
        if name not in func_map:
            features.append(0.0)
            continue
        func = func_map[name]
        value = func(mol, avg=True) if name == "Ipc" else func(mol)
        features.append(value)
    return np.asarray(features)


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_smiles(smiles_list, fixed_names, func_map):
    """Convert a list of SMILES to a descriptor matrix.

    Returns
    -------
    X : ndarray of shape (n_valid, n_descriptors)
    valid_mask : boolean array – True for rows kept
    invalid_indices : integer array – original indices that were dropped
    """
    X = np.array([
        _compute_descriptors(Chem.MolFromSmiles(smi), fixed_names, func_map)
        for smi in smiles_list
    ])
    valid_mask = ~np.isnan(X).any(axis=1)
    invalid_idx = np.where(~valid_mask)[0]
    return X[valid_mask], valid_mask, invalid_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict molecular activity from SMILES via a pre-trained SVM model."
    )
    parser.add_argument(
        "--input", "-i", default="smile.csv",
        help="Input CSV file. The first column must contain SMILES strings. (default: smile.csv)",
    )
    parser.add_argument(
        "--output", "-o", default="result.csv",
        help="Output CSV file with prediction results. (default: result.csv)",
    )
    parser.add_argument(
        "--model", "-m", default="SVMRDKitDesc.pickle",
        help="Path to the trained SVM model pickle file. (default: SVMRDKitDesc.pickle)",
    )
    parser.add_argument(
        "--descriptors", "-d", default="descriptor_columns.json",
        help="Path to the descriptor column name list (JSON). (default: descriptor_columns.json)",
    )
    args = parser.parse_args()

    # 1. Load data
    df = pd.read_csv(args.input)
    smiles_list = df.iloc[:, 0].tolist()
    print(f"Loaded {len(smiles_list)} molecules from '{args.input}'")

    # 2. Compute descriptors
    fixed_names, func_map = _load_descriptor_config(args.descriptors)
    X, valid_mask, invalid_idx = process_smiles(smiles_list, fixed_names, func_map)

    if len(invalid_idx) > 0:
        print(f"Warning: {len(invalid_idx)} molecules dropped (invalid descriptors), "
              f"row indices: {invalid_idx.tolist()}")

    # 3. Load model & predict
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # 4. Save results
    df_result = df[valid_mask].copy()
    df_result["predict"] = predictions
    df_result["predict_proba"] = probabilities
    df_result.to_csv(args.output, index=False)
    print(f"Results saved to '{args.output}' ({len(df_result)} molecules)")


if __name__ == "__main__":
    main()
