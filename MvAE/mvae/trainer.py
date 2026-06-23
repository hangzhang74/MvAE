"""High-level training interface for MvAE."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import load_real_data
from .metrics import evaluate_clustering_only, extract_reconstructed_matrix_and_cluster, is_better_cluster_metric
from .model import MvAEModel
from .utils import parse_hidden_dims, resolve_device, set_seed


@dataclass
class MvAEConfig:
    """Configuration for MvAE training and output extraction."""

    view1: str
    view2: str
    labels: str
    out_dir: str
    save_path: str = ""

    cell_col: str = "Cell"
    label_col: str = "Label"

    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    recon_weight: float = 1.0
    class_weight: float = 1.0
    grad_clip: Optional[float] = 5.0
    seed: int = 42
    num_workers: int = 0

    hidden_dims: str | tuple[int, ...] = "1024,512,256"
    latent_dim: int = 128
    dropout_rate: float = 0.2
    attn_dim: int = 128
    attn_dropout: float = 0.2
    attn_layers: int = 2
    decoder_output_activation: str | None = "sigmoid"

    device: str = "auto"

    cluster_feature: str = "x1_recon"
    kmeans_n_init: int = 50
    best_cluster_metric: str = "CompositeScore"
    cluster_eval_every: int = 1
    skip_clustering: bool = False


class MvAE:
    """
    Encapsulated MvAE trainer.

    Example:
        from mvae import MvAE, MvAEConfig

        cfg = MvAEConfig(
            view1="final_raw_HVG_counts_MinMax_cell_by_gene.csv",
            view2="final_GLNE_MinMax_cell_by_gene.csv",
            labels="cell_labels.csv",
            out_dir="MvAE_output",
        )
        runner = MvAE(cfg)
        runner.fit()
    """

    def __init__(self, config: MvAEConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.model = None
        self.optimizer = None
        self.data = None
        self.history: list[dict] = []
        self.cluster_history: list[dict] = []
        self.best_epoch = None
        self.best_metric_value = None
        self.best_cluster_metrics = None
        self.save_path = self._resolve_save_path()

    def _resolve_save_path(self) -> str:
        cfg = self.config
        if cfg.save_path is None or str(cfg.save_path).strip() == "":
            return os.path.join(
                cfg.out_dir,
                f"MvAE_attn_layers{cfg.attn_layers}_best_{cfg.best_cluster_metric}_{cfg.cluster_feature}.pt",
            )
        return cfg.save_path

    def _load_data(self) -> None:
        cfg = self.config
        self.data = load_real_data(
            view1_path=cfg.view1,
            view2_path=cfg.view2,
            label_path=cfg.labels,
            cell_col=cfg.cell_col,
            label_col=cfg.label_col,
        )

    def _build_model(self) -> None:
        if self.data is None:
            self._load_data()
        _, _, _, input_dim, num_classes, _, _, _ = self.data
        cfg = self.config
        hidden_dims = parse_hidden_dims(cfg.hidden_dims)
        decoder_activation = None if cfg.decoder_output_activation == "none" else cfg.decoder_output_activation

        self.model = MvAEModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.latent_dim,
            num_classes=num_classes,
            dropout_rate=cfg.dropout_rate,
            attn_dim=cfg.attn_dim,
            attn_dropout=cfg.attn_dropout,
            attn_layers=cfg.attn_layers,
            decoder_output_activation=decoder_activation,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        print("[INFO] Device:", self.device)
        print("[INFO] input_dim:", input_dim)
        print("[INFO] num_classes:", num_classes)
        print("[INFO] Model hidden_dims:", hidden_dims)
        print("[INFO] Model latent_dim:", cfg.latent_dim)
        print("[INFO] Model attn_layers:", cfg.attn_layers)

    @staticmethod
    def compute_loss(outputs, x1, x2, y, recon_weight: float = 1.0, class_weight: float = 1.0):
        x1_recon = outputs["x1_recon"]
        x2_recon = outputs["x2_recon"]
        logits = outputs["logits"]

        loss_recon1 = F.mse_loss(x1_recon, x1)
        loss_recon2 = F.mse_loss(x2_recon, x2)
        loss_recon = loss_recon1 + loss_recon2
        loss_class = F.cross_entropy(logits, y)
        loss_total = recon_weight * loss_recon + class_weight * loss_class

        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_recon": loss_recon.item(),
            "loss_recon1": loss_recon1.item(),
            "loss_recon2": loss_recon2.item(),
            "loss_class": loss_class.item(),
        }
        return loss_total, loss_dict

    def _make_train_loader(self) -> DataLoader:
        cfg = self.config
        x1, x2, y, *_ = self.data
        train_dataset = TensorDataset(x1, x2, y)
        return DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def train_one_epoch(self, loader: DataLoader) -> dict:
        cfg = self.config
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_class = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x1, batch_x2, batch_y in loader:
            batch_x1 = batch_x1.to(self.device)
            batch_x2 = batch_x2.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x1, batch_x2, return_attention=False)
            loss, loss_dict = self.compute_loss(
                outputs=outputs,
                x1=batch_x1,
                x2=batch_x2,
                y=batch_y,
                recon_weight=cfg.recon_weight,
                class_weight=cfg.class_weight,
            )

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
            self.optimizer.step()

            batch_size = batch_x1.size(0)
            total_loss += loss_dict["loss_total"] * batch_size
            total_recon += loss_dict["loss_recon"] * batch_size
            total_class += loss_dict["loss_class"] * batch_size
            pred = outputs["logits"].argmax(dim=1)
            total_correct += (pred == batch_y).sum().item()
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "loss_recon": total_recon / total_samples,
            "loss_class": total_class / total_samples,
            "acc": total_correct / total_samples,
        }

    def fit(self):
        """Train MvAE, save best checkpoint, histories, and final outputs."""
        cfg = self.config
        set_seed(cfg.seed)
        os.makedirs(cfg.out_dir, exist_ok=True)

        self._load_data()
        self._build_model()
        train_loader = self._make_train_loader()

        x1, x2, y, input_dim, num_classes, label_encoder, _, _ = self.data
        hidden_dims = parse_hidden_dims(cfg.hidden_dims)
        decoder_activation = None if cfg.decoder_output_activation == "none" else cfg.decoder_output_activation

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_recon": train_metrics["loss_recon"],
                "train_class": train_metrics["loss_class"],
                "train_acc": train_metrics["acc"],
            }

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_recon={train_metrics['loss_recon']:.4f} "
                f"train_class={train_metrics['loss_class']:.4f} "
                f"train_acc={train_metrics['acc']:.4f}"
            )

            if (epoch % cfg.cluster_eval_every == 0) or (epoch == cfg.epochs):
                cluster_metrics, _ = evaluate_clustering_only(
                    model=self.model,
                    x1=x1,
                    x2=x2,
                    y=y,
                    label_encoder=label_encoder,
                    device=self.device,
                    batch_size=cfg.batch_size,
                    use_feature=cfg.cluster_feature,
                    kmeans_n_init=cfg.kmeans_n_init,
                    seed=cfg.seed,
                )
                cluster_row = {"epoch": epoch, **cluster_metrics}
                self.cluster_history.append(cluster_row)
                row.update(cluster_metrics)

                print(
                    f"[Epoch {epoch:03d} CLUSTER] "
                    f"feature={cfg.cluster_feature} "
                    f"ARI={cluster_metrics['ARI']:.4f} "
                    f"NMI={cluster_metrics['NMI']:.4f} "
                    f"Purity={cluster_metrics['Purity']:.4f} "
                    f"Silhouette={cluster_metrics['Silhouette']:.4f} "
                    f"DBI={cluster_metrics['DBI']:.4f} "
                    f"CompositeScore={cluster_metrics['CompositeScore']:.4f}"
                )

                current_metric_value = cluster_metrics[cfg.best_cluster_metric]
                if is_better_cluster_metric(cfg.best_cluster_metric, current_metric_value, self.best_metric_value):
                    self.best_metric_value = current_metric_value
                    self.best_epoch = epoch
                    self.best_cluster_metrics = cluster_metrics.copy()
                    self.save_checkpoint(
                        input_dim=input_dim,
                        num_classes=num_classes,
                        classes=label_encoder.classes_.tolist(),
                        hidden_dims=hidden_dims,
                        decoder_activation=decoder_activation,
                    )
                    print(
                        f"[OK] Best model updated by {cfg.best_cluster_metric}: "
                        f"{self.best_metric_value:.6f} at epoch {self.best_epoch}"
                    )

            self.history.append(row)

        self.save_history()
        print("[DONE] Training finished.")
        print(f"[BEST] metric={cfg.best_cluster_metric}")
        print(f"[BEST] value={self.best_metric_value}")
        print(f"[BEST] epoch={self.best_epoch}")
        print("[BEST] checkpoint saved:")
        print(self.save_path)

        if not cfg.skip_clustering:
            self.load_best_checkpoint()
            self.extract_outputs()
            print("[DONE] Best-model reconstruction and clustering finished.")
        else:
            print("[INFO] Clustering skipped because skip_clustering=True.")
        return self

    def save_checkpoint(self, input_dim: int, num_classes: int, classes, hidden_dims, decoder_activation) -> None:
        cfg = self.config
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "classes": classes,
                "attn_layers": cfg.attn_layers,
                "latent_dim": cfg.latent_dim,
                "hidden_dims": hidden_dims,
                "dropout_rate": cfg.dropout_rate,
                "attn_dim": cfg.attn_dim,
                "attn_dropout": cfg.attn_dropout,
                "decoder_output_activation": decoder_activation,
                "view1_path": cfg.view1,
                "view2_path": cfg.view2,
                "label_path": cfg.labels,
                "cell_col": cfg.cell_col,
                "label_col": cfg.label_col,
                "best_epoch": self.best_epoch,
                "best_cluster_metric_name": cfg.best_cluster_metric,
                "best_cluster_metric_value": self.best_metric_value,
                "best_cluster_metrics": self.best_cluster_metrics,
                "cluster_feature": cfg.cluster_feature,
                "kmeans_n_init": cfg.kmeans_n_init,
                "seed": cfg.seed,
                "config": asdict(cfg),
            },
            self.save_path,
        )

    def save_history(self) -> None:
        cfg = self.config
        history_path = os.path.join(cfg.out_dir, "training_history_with_clustering.csv")
        pd.DataFrame(self.history).to_csv(history_path, index=False)
        print("[OK] Training history with clustering saved:")
        print(history_path)

        cluster_history_path = os.path.join(cfg.out_dir, "epoch_clustering_metrics.csv")
        pd.DataFrame(self.cluster_history).to_csv(cluster_history_path, index=False)
        print("[OK] Epoch clustering metrics saved:")
        print(cluster_history_path)

    def load_best_checkpoint(self) -> None:
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("[OK] Loaded best clustering checkpoint:")
            print(self.save_path)
            print("[INFO] Best epoch:", checkpoint.get("best_epoch"))
            print("[INFO] Best metric:", checkpoint.get("best_cluster_metric_name"))
            print("[INFO] Best metric value:", checkpoint.get("best_cluster_metric_value"))
        else:
            print("[WARN] Best checkpoint not found. Using current final model.")

    def extract_outputs(self):
        cfg = self.config
        x1, x2, y, _, _, label_encoder, cell_names, gene_names = self.data
        return extract_reconstructed_matrix_and_cluster(
            model=self.model,
            x1=x1,
            x2=x2,
            y=y,
            cell_names=cell_names,
            gene_names=gene_names,
            label_encoder=label_encoder,
            device=self.device,
            batch_size=cfg.batch_size,
            out_dir=cfg.out_dir,
            use_feature=cfg.cluster_feature,
            kmeans_n_init=cfg.kmeans_n_init,
            seed=cfg.seed,
        )
