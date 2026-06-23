"""Data loading utilities for MvAE."""

from __future__ import annotations

import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def load_real_data(
    view1_path: str,
    view2_path: str,
    label_path: str,
    cell_col: str = "Cell",
    label_col: str = "Label",
):
    """
    Load two-view single-cell matrices and labels.

    Required format:
        view1_path: CSV, rows = cells, columns = genes/features
        view2_path: CSV, rows = cells, columns = genes/features
        label_path: CSV, must contain cell_col and label_col
    """
    for path_name, path_value in {
        "view1_path": view1_path,
        "view2_path": view2_path,
        "label_path": label_path,
    }.items():
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"{path_name} does not exist: {path_value}")

    x1_df = pd.read_csv(view1_path, index_col=0)
    x2_df = pd.read_csv(view2_path, index_col=0)
    x1_df.index = x1_df.index.astype(str)
    x2_df.index = x2_df.index.astype(str)

    x1_df = x1_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x2_df = x2_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print("[INFO] View 1 path:", view1_path)
    print("[INFO] View 2 path:", view2_path)
    print("[INFO] View 1 shape:", x1_df.shape)
    print("[INFO] View 2 shape:", x2_df.shape)

    if x1_df.shape[1] != x2_df.shape[1]:
        raise ValueError(
            "View 1 and View 2 must have the same number of features. "
            f"Got {x1_df.shape[1]} and {x2_df.shape[1]}."
        )

    label_df = pd.read_csv(label_path)
    print("[INFO] Label path:", label_path)
    print("[INFO] Label columns:", label_df.columns.tolist())

    if cell_col not in label_df.columns:
        raise ValueError(f"Label file must contain cell column: {cell_col}")
    if label_col not in label_df.columns:
        raise ValueError(f"Label file must contain label column: {label_col}")

    label_df[cell_col] = label_df[cell_col].astype(str)
    label_df[label_col] = label_df[label_col].astype(str)
    label_df = label_df[[cell_col, label_col]].dropna()
    label_df = label_df.drop_duplicates(subset=[cell_col])
    label_df = label_df.set_index(cell_col)

    common_cells = x1_df.index.intersection(x2_df.index).intersection(label_df.index)
    print("[INFO] Number of common cells:", len(common_cells))
    if len(common_cells) == 0:
        raise ValueError("No common cells among view1, view2, and label file.")

    x1_df = x1_df.loc[common_cells]
    x2_df = x2_df.loc[common_cells]
    label_df = label_df.loc[common_cells]

    common_features = x1_df.columns.intersection(x2_df.columns)
    if len(common_features) == 0:
        raise ValueError("No common features between View 1 and View 2.")
    if len(common_features) != x1_df.shape[1] or len(common_features) != x2_df.shape[1]:
        print("[WARN] View 1 and View 2 feature columns are not identical.")
        print("[WARN] Using common features only:", len(common_features))

    x1_df = x1_df.loc[:, common_features]
    x2_df = x2_df.loc[:, common_features]

    print("[INFO] View 1 after alignment:", x1_df.shape)
    print("[INFO] View 2 after alignment:", x2_df.shape)
    print("[INFO] Labels after alignment:", label_df.shape)

    label_encoder = LabelEncoder()
    y_np = label_encoder.fit_transform(label_df[label_col].astype(str).values)

    print("[INFO] Classes:")
    for i, cls in enumerate(label_encoder.classes_):
        print(i, cls)

    x1 = torch.tensor(x1_df.values, dtype=torch.float32)
    x2 = torch.tensor(x2_df.values, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    input_dim = x1.shape[1]
    num_classes = len(label_encoder.classes_)
    cell_names = common_cells.astype(str).tolist()
    gene_names = x1_df.columns.astype(str).tolist()

    return x1, x2, y, input_dim, num_classes, label_encoder, cell_names, gene_names
