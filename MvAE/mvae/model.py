"""Neural network components for MvAE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_encoder(input_dim: int, hidden_dims, latent_dim: int, dropout_rate: float = 0.2) -> nn.Sequential:
    """Create an encoder: input_dim -> hidden_dims -> latent_dim."""
    layers = []
    if hidden_dims is None or len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, latent_dim))
        return nn.Sequential(*layers)

    hidden_dims = list(hidden_dims)
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(hidden_dims[-1], latent_dim))
    return nn.Sequential(*layers)


def create_decoder(latent_dim: int, hidden_dims, output_dim: int, output_activation: str | None = "sigmoid") -> nn.Sequential:
    """Create a decoder: latent_dim -> reversed hidden_dims -> output_dim."""
    layers = []
    if hidden_dims is None or len(hidden_dims) == 0:
        layers.append(nn.Linear(latent_dim, output_dim))
    else:
        hidden_dims = list(hidden_dims)
        decoder_hidden_dims = hidden_dims[::-1]
        dims = [latent_dim] + decoder_hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(decoder_hidden_dims[-1], output_dim))

    if output_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif output_activation == "relu":
        layers.append(nn.ReLU())
    elif output_activation is None:
        pass
    else:
        raise ValueError("output_activation must be 'sigmoid', 'relu', or None.")
    return nn.Sequential(*layers)


class SameFeatureSampleAttentionLayer(nn.Module):
    """
    Same-feature sample-level attention.

    Input: z with shape [batch_size, latent_dim]. For each latent feature dimension,
    cells in the mini-batch are treated as tokens and sample-sample attention is computed.
    """

    def __init__(
        self,
        latent_dim: int,
        attn_dim: int = 32,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.attn_dim = attn_dim
        self.use_residual = use_residual
        self.query = nn.Linear(1, attn_dim)
        self.key = nn.Linear(1, attn_dim)
        self.value = nn.Linear(1, attn_dim)
        self.out_proj = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()

    def forward(self, z: torch.Tensor, return_attention: bool = False):
        batch_size, dim = z.shape
        if dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, but got D={dim}.")

        z_feature_sample = z.transpose(0, 1).unsqueeze(-1)  # [D, B, 1]
        query = self.query(z_feature_sample)
        key = self.key(z_feature_sample)
        value = self.value(z_feature_sample)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        attn_matrix = F.softmax(attn_scores, dim=-1)
        attn_matrix = self.dropout(attn_matrix)

        z_att = torch.matmul(attn_matrix, value)
        z_att = self.out_proj(z_att).squeeze(-1).transpose(0, 1)  # [B, D]
        z_out = z + z_att if self.use_residual else z_att
        z_out = self.norm(z_out)

        if return_attention:
            return z_out, attn_matrix.detach()
        return z_out, None


class SameFeatureSampleAttentionStack(nn.Module):
    """Stacked same-feature sample-level attention."""

    def __init__(
        self,
        latent_dim: int,
        attn_dim: int = 32,
        dropout_rate: float = 0.1,
        attn_layers: int = 1,
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if attn_layers < 1:
            raise ValueError("attn_layers must be >= 1.")
        self.attn_layers = attn_layers
        self.layers = nn.ModuleList(
            [
                SameFeatureSampleAttentionLayer(
                    latent_dim=latent_dim,
                    attn_dim=attn_dim,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                    use_layernorm=use_layernorm,
                )
                for _ in range(attn_layers)
            ]
        )

    def forward(self, z: torch.Tensor, return_attention: bool = False, return_all_attention: bool = False):
        attn_list = []
        last_attn = None
        for layer in self.layers:
            z, attn = layer(z, return_attention=return_attention)
            if return_attention:
                last_attn = attn
                if return_all_attention:
                    attn_list.append(attn)
        if return_attention:
            return z, attn_list if return_all_attention else last_attn
        return z, None


class MultiViewSameFeatureAttentionAE(nn.Module):
    """Multi-view autoencoder with stacked same-feature sample-level attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(1024, 512, 256),
        latent_dim: int = 128,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        attn_dim: int = 32,
        attn_dropout: float = 0.1,
        attn_layers: int = 2,
        decoder_output_activation: str | None = "sigmoid",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.attn_layers = attn_layers

        self.encoder1 = create_encoder(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.encoder2 = create_encoder(input_dim, hidden_dims, latent_dim, dropout_rate)

        self.attn1 = SameFeatureSampleAttentionStack(
            latent_dim=latent_dim,
            attn_dim=attn_dim,
            dropout_rate=attn_dropout,
            attn_layers=attn_layers,
            use_residual=True,
            use_layernorm=True,
        )
        self.attn2 = SameFeatureSampleAttentionStack(
            latent_dim=latent_dim,
            attn_dim=attn_dim,
            dropout_rate=attn_dropout,
            attn_layers=attn_layers,
            use_residual=True,
            use_layernorm=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(latent_dim),
        )

        self.decoder1 = create_decoder(latent_dim, hidden_dims, input_dim, decoder_output_activation)
        self.decoder2 = create_decoder(latent_dim, hidden_dims, input_dim, decoder_output_activation)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False, return_all_attention: bool = False):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        z1_att, attn1 = self.attn1(z1, return_attention=return_attention, return_all_attention=return_all_attention)
        z2_att, attn2 = self.attn2(z2, return_attention=return_attention, return_all_attention=return_all_attention)

        z_fused = self.fusion(torch.cat([z1_att, z2_att], dim=1))
        x1_recon = self.decoder1(z_fused)
        x2_recon = self.decoder2(z_fused)
        logits = self.classifier(z_fused)

        outputs = {
            "z1": z1,
            "z2": z2,
            "z1_att": z1_att,
            "z2_att": z2_att,
            "z_fused": z_fused,
            "x1_recon": x1_recon,
            "x2_recon": x2_recon,
            "logits": logits,
        }
        if return_attention:
            outputs["attn1"] = attn1
            outputs["attn2"] = attn2
        return outputs


MvAEModel = MultiViewSameFeatureAttentionAE
