"""Clustering metrics and output extraction for MvAE."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from torch.utils.data import DataLoader, TensorDataset


def purity_score(y_true, y_pred) -> float:
    """Compute clustering purity."""
    contingency = pd.crosstab(y_pred, y_true)
    return float(np.sum(np.max(contingency.values, axis=1)) / np.sum(contingency.values))


def composite_cluster_score(ari: float, nmi: float, purity: float, dbi: float) -> float:
    """Composite score: ARI + NMI + Purity + (1 - DBI)."""
    if dbi is None or np.isnan(dbi):
        return np.nan
    return float(ari + nmi + purity + (1.0 - dbi))


def is_better_cluster_metric(metric_name: str, current_value, best_value) -> bool:
    """Return True when current clustering metric improves over the previous best."""
    if current_value is None:
        return False
    if np.isnan(current_value):
        return False
    if best_value is None:
        return True
    if metric_name == "DBI":
        return current_value < best_value
    return current_value > best_value


@torch.no_grad()
def evaluate_clustering_only(
    model,
    x1,
    x2,
    y,
    label_encoder,
    device,
    batch_size: int = 128,
    use_feature: str = "x1_recon",
    kmeans_n_init: int = 50,
    seed: int = 42,
):
    """Evaluate clustering metrics without saving reconstructed matrices."""
    model.eval()
    all_dataset = TensorDataset(x1, x2, y)
    all_loader = DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    feature_list = []
    y_list = []
    for batch_x1, batch_x2, batch_y in all_loader:
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        outputs = model(batch_x1, batch_x2, return_attention=False)

        if use_feature == "x1_recon":
            feat = outputs["x1_recon"]
        elif use_feature == "x2_recon":
            feat = outputs["x2_recon"]
        elif use_feature == "z_fused":
            feat = outputs["z_fused"]
        else:
            raise ValueError("use_feature must be 'x1_recon', 'x2_recon', or 'z_fused'.")

        feature_list.append(feat.detach().cpu().numpy())
        y_list.append(batch_y.numpy())

    cluster_input = np.vstack(feature_list)
    y_np = np.concatenate(y_list)
    n_clusters = len(label_encoder.classes_)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=kmeans_n_init)
    cluster_pred = kmeans.fit_predict(cluster_input)

    ari = adjusted_rand_score(y_np, cluster_pred)
    nmi = normalized_mutual_info_score(y_np, cluster_pred)
    purity = purity_score(y_np, cluster_pred)
    if len(np.unique(cluster_pred)) > 1:
        silhouette = silhouette_score(cluster_input, cluster_pred)
        dbi = davies_bouldin_score(cluster_input, cluster_pred)
    else:
        silhouette = np.nan
        dbi = np.nan

    metrics = {
        "cluster_feature": use_feature,
        "n_cells": len(y_np),
        "n_clusters": n_clusters,
        "ARI": ari,
        "NMI": nmi,
        "Purity": purity,
        "Silhouette": silhouette,
        "DBI": dbi,
        "CompositeScore": composite_cluster_score(ari, nmi, purity, dbi),
    }
    return metrics, cluster_pred


@torch.no_grad()
def extract_reconstructed_matrix_and_cluster(
    model,
    x1,
    x2,
    y,
    cell_names,
    gene_names,
    label_encoder,
    device,
    batch_size: int = 128,
    out_dir: str = "./outputs",
    use_feature: str = "x1_recon",
    kmeans_n_init: int = 50,
    seed: int = 42,
):
    """Save MvAE reconstructed matrices, fused latent representation, KMeans labels, and metrics."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_dataset = TensorDataset(x1, x2, y)
    all_loader = DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    x1_recon_list, x2_recon_list, z_fused_list, y_list = [], [], [], []
    for batch_x1, batch_x2, batch_y in all_loader:
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        outputs = model(batch_x1, batch_x2, return_attention=False)
        x1_recon_list.append(outputs["x1_recon"].detach().cpu().numpy())
        x2_recon_list.append(outputs["x2_recon"].detach().cpu().numpy())
        z_fused_list.append(outputs["z_fused"].detach().cpu().numpy())
        y_list.append(batch_y.numpy())

    x1_recon_np = np.vstack(x1_recon_list)
    x2_recon_np = np.vstack(x2_recon_list)
    z_fused_np = np.vstack(z_fused_list)
    y_np = np.concatenate(y_list)

    x1_recon_path = os.path.join(out_dir, "MvAE_reconstructed_expression_matrix_cell_by_gene.csv")
    pd.DataFrame(x1_recon_np, index=cell_names, columns=gene_names).to_csv(x1_recon_path)
    print("[OK] MvAE reconstructed expression matrix saved:")
    print(x1_recon_path)

    x2_recon_path = os.path.join(out_dir, "MvAE_reconstructed_GLNE_matrix_cell_by_gene.csv")
    pd.DataFrame(x2_recon_np, index=cell_names, columns=gene_names).to_csv(x2_recon_path)
    print("[OK] MvAE reconstructed GLNE matrix saved:")
    print(x2_recon_path)

    z_fused_path = os.path.join(out_dir, "MvAE_fused_latent_representation.csv")
    z_columns = [f"z{i + 1}" for i in range(z_fused_np.shape[1])]
    pd.DataFrame(z_fused_np, index=cell_names, columns=z_columns).to_csv(z_fused_path)
    print("[OK] MvAE fused latent representation saved:")
    print(z_fused_path)

    if use_feature == "x1_recon":
        cluster_input = x1_recon_np
        cluster_name = "x1_recon"
    elif use_feature == "z_fused":
        cluster_input = z_fused_np
        cluster_name = "z_fused"
    else:
        raise ValueError("use_feature must be 'x1_recon' or 'z_fused'.")

    n_clusters = len(label_encoder.classes_)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=kmeans_n_init)
    cluster_pred = kmeans.fit_predict(cluster_input)

    ari = adjusted_rand_score(y_np, cluster_pred)
    nmi = normalized_mutual_info_score(y_np, cluster_pred)
    purity = purity_score(y_np, cluster_pred)
    if len(np.unique(cluster_pred)) > 1:
        silhouette = silhouette_score(cluster_input, cluster_pred)
        dbi = davies_bouldin_score(cluster_input, cluster_pred)
    else:
        silhouette = np.nan
        dbi = np.nan

    metrics = {
        "cluster_feature": cluster_name,
        "n_cells": len(y_np),
        "n_clusters": n_clusters,
        "ARI": ari,
        "NMI": nmi,
        "Purity": purity,
        "Silhouette": silhouette,
        "DBI": dbi,
        "CompositeScore": composite_cluster_score(ari, nmi, purity, dbi),
    }

    print("\n[CLUSTERING RESULTS]")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    metrics_path = os.path.join(out_dir, f"MvAE_KMeans_clustering_metrics_{cluster_name}.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("[OK] Clustering metrics saved:")
    print(metrics_path)

    cluster_path = os.path.join(out_dir, f"MvAE_KMeans_cluster_labels_{cluster_name}.csv")
    true_labels = label_encoder.inverse_transform(y_np)
    pd.DataFrame(
        {
            "Cell": cell_names,
            "true_label_id": y_np,
            "true_label": true_labels,
            "kmeans_cluster": cluster_pred,
        }
    ).to_csv(cluster_path, index=False)
    print("[OK] Cluster labels saved:")
    print(cluster_path)

    return {
        "x1_recon": x1_recon_np,
        "x2_recon": x2_recon_np,
        "z_fused": z_fused_np,
        "cluster_pred": cluster_pred,
        "metrics": metrics,
        "paths": {
            "x1_recon": x1_recon_path,
            "x2_recon": x2_recon_path,
            "z_fused": z_fused_path,
            "metrics": metrics_path,
            "clusters": cluster_path,
        },
    }
