"""Command-line interface for MvAE."""

from __future__ import annotations

import argparse

from .trainer import MvAE, MvAEConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train MvAE: a multi-view same-feature sample-level attention autoencoder. "
            "No validation/test split is used."
        )
    )

    parser.add_argument("--view1", type=str, required=True, help="Path to View 1 CSV. Rows = cells, columns = genes/features.")
    parser.add_argument("--view2", type=str, required=True, help="Path to View 2 CSV. Rows = cells, columns = genes/features.")
    parser.add_argument("--labels", type=str, required=True, help="Path to label CSV. Must contain --cell_col and --label_col.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--save_path", type=str, default="", help="Model checkpoint path. If empty, save under out_dir.")

    parser.add_argument("--cell_col", type=str, default="Cell", help="Cell ID column name in label CSV.")
    parser.add_argument("--label_col", type=str, default="Label", help="Cell label column name in label CSV.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--class_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="1024,512,256",
        help='Hidden dimensions, for example "1024,512,256". Use "" for no hidden layer.',
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument(
        "--decoder_output_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "relu", "none"],
        help="Decoder output activation.",
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cluster_feature", type=str, default="x1_recon", choices=["x1_recon", "z_fused"])
    parser.add_argument("--kmeans_n_init", type=int, default=50)
    parser.add_argument(
        "--best_cluster_metric",
        type=str,
        default="CompositeScore",
        choices=["CompositeScore", "ARI", "NMI", "Purity", "Silhouette", "DBI"],
    )
    parser.add_argument("--cluster_eval_every", type=int, default=1)
    parser.add_argument("--skip_clustering", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    config = MvAEConfig(**vars(args))
    MvAE(config).fit()


if __name__ == "__main__":
    main()
