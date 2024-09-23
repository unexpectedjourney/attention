import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, path_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = path_size
        self.n_patches = (img_size // path_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, path_size, stride=path_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        assert dim == self.dim, "dim != self.dim"

        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        attn_score = ((q @ k_t) * self.scale).softmax(dim=-1)
        attn_score = self.attn_drop(attn_score)

        weighted_average = attn_score @ v
        weighted_average = weighted_average.transpose(1, 2)
        weighted_average = weighted_average.flatten(2)

        x = self.proj(weighted_average)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self, dim, n_heads, mlp_ration=4.0, qkv_bias=True, attn_p=0., p=0.,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.norm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.hidden_features = int(dim * mlp_ration)
        self.mlp = MLP(dim, self.hidden_features, dim)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=386, patch_size=16, in_channels=3,
        n_classes=1000, embed_dim=768, depth=12, n_heads=12,
        mlp_ration=4.0, qkv_bias=True, attn_p=0., p=0.,
    ):
        super().__init__()

        self.patch_emb = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_emb.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p)
        self.encoder_blocks = [
            EncoderBlock(
                embed_dim, n_heads, mlp_ration, qkv_bias, attn_p, p
            ) for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_sampels = x.shape[0]
        x = self.patch_emb(x)

        cls_token = self.cls_token.expand(
            n_sampels, -1, -1
        )
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)

        clf_token_final = x[:, 0]
        x = self.head(clf_token_final)

        return x
