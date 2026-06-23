# MvAE

Encapsulated version of the original single-file MvAE script. It keeps the original training logic, model structure, clustering evaluation, checkpoint selection, and output file names, but reorganizes the code into a reusable Python package.

## Package structure

```text
MvAE/
├── mvae/
│   ├── __init__.py
│   ├── cli.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── run_mvae.py
├── requirements.txt
└── README.md
```

## Input format

- `view1`: CSV, rows = cells, columns = genes/features.
- `view2`: CSV, rows = cells, columns = genes/features.
- `labels`: CSV containing cell IDs and labels.
- Default label columns are `Cell` and `Label`. Change them using `--cell_col` and `--label_col`.

## Command-line usage

```bash
cd /path/to/MvAE

python run_mvae.py \
  --view1 /home/zhanghang/GSE132188/Seurat_CellEnergy_full_pipeline/final_raw_HVG_counts_MinMax_cell_by_gene.csv \
  --view2 /home/zhanghang/GSE132188/Seurat_CellEnergy_full_pipeline/final_GLNE_MinMax_cell_by_gene.csv \
  --labels /home/zhanghang/GSE132188/Seurat_CellEnergy_full_pipeline/cell_labels_after_QC.csv \
  --out_dir /home/zhanghang/GSE132188/Seurat_CellEnergy_full_pipeline/MvAE_output \
  --cell_col Cell \
  --label_col Label \
  --epochs 100 \
  --batch_size 256 \
  --cluster_feature x1_recon \
  --best_cluster_metric CompositeScore \
  --device auto
```

For GSE158490/GSE158493-like data, replace the paths and label column name as needed.

## Python usage

```python
from mvae import MvAE, MvAEConfig

config = MvAEConfig(
    view1="final_raw_HVG_counts_MinMax_cell_by_gene.csv",
    view2="final_GLNE_MinMax_cell_by_gene.csv",
    labels="cell_labels.csv",
    out_dir="MvAE_output",
    cell_col="Cell",
    label_col="Label",
    epochs=100,
    batch_size=256,
    cluster_feature="x1_recon",
    best_cluster_metric="CompositeScore",
)

runner = MvAE(config)
runner.fit()
```

## Main outputs

The following output names are preserved:

- `MvAE_reconstructed_expression_matrix_cell_by_gene.csv`
- `MvAE_reconstructed_GLNE_matrix_cell_by_gene.csv`
- `MvAE_fused_latent_representation.csv`
- `MvAE_KMeans_clustering_metrics_x1_recon.csv` or `MvAE_KMeans_clustering_metrics_z_fused.csv`
- `MvAE_KMeans_cluster_labels_x1_recon.csv` or `MvAE_KMeans_cluster_labels_z_fused.csv`
- `training_history_with_clustering.csv`
- `epoch_clustering_metrics.csv`
- best checkpoint: `MvAE_attn_layers{K}_best_{metric}_{feature}.pt`

## Notes

- Default model: `2000 -> 1024 -> 512 -> 256 -> 128` encoders, two-view reconstruction, classifier head, stacked same-feature sample-level attention, and fused latent representation.
- Default reconstruction output activation is `sigmoid`, which matches Min-Max normalized inputs in `[0, 1]`.
- No validation/test split is used, matching the original script.
