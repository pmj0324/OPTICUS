"""Vision Transformer model for OPTICUS.
OPTICUS용 Vision Transformer 모델.
"""
import torch
import torch.nn as nn


class ViT50_3block(nn.Module):
    """Vision Transformer with patch size 50 and 3 transformer blocks.
    패치 크기 50, 3개의 transformer 블록을 가진 Vision Transformer.
    
    Args / 인자:
        img_size: Input image resolution (default: 500x500)
                 입력 이미지 해상도 (기본값: 500x500)
        patch_size: Size of each patch (default: 50x50)
                   각 패치의 크기 (기본값: 50x50)
        embed_dim: Patch embedding dimension (default: 256)
                  패치 임베딩 차원 (기본값: 256)
        depth: Number of Transformer Encoder layers (default: 3)
              Transformer Encoder 레이어 수 (기본값: 3)
        num_heads: Number of multi-head attention heads (default: 8)
                  멀티헤드 어텐션 헤드 수 (기본값: 8)
        mlp_dim: Hidden dimension in Transformer MLP (default: 512)
                Transformer MLP의 숨겨진 차원 (기본값: 512)
        num_classes: Number of output nodes (default: 1 for regression)
                    출력 노드 수 (회귀용 기본값: 1)
    """
    
    def __init__(self,
                 img_size=500,
                 patch_size=50,
                 embed_dim=256,
                 depth=3,
                 num_heads=8,
                 mlp_dim=512,
                 num_classes=1):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (img_size // patch_size) ** 2  # (500/50)² = 10×10 = 100

        # Patch Embedding: Conv2d(1→embed_dim, kernel=patch_size, stride=patch_size)
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class Token: learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding: (1, num_patches+1, embed_dim)
        # Sequence length = num_patches(100) + cls token(1) = 101
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder: depth layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation='gelu',
            batch_first=False  # expects (seq_len, batch, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Regression Head: Transformer CLS output(embed_dim) → Linear(embed_dim→num_classes)
        self.head = nn.Linear(embed_dim, num_classes)

        # Parameter initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """Forward pass.
        순전파.
        
        Args / 인자:
            x: Input tensor of shape (B, 1, 500, 500)
               입력 텐서, 형태 (B, 1, 500, 500)
            
        Returns / 반환:
            Output tensor of shape (B,) for regression
            회귀용 출력 텐서, 형태 (B,)
        """
        B = x.size(0)

        # (1) Patch Embedding
        # Input (B,1,500,500) → Conv2d → (B, embed_dim, 10, 10)
        x = self.patch_embed(x)

        # (2) Flatten & Transpose
        # (B, embed_dim, 10, 10) → (B, embed_dim, 100) → (B, 100, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # (3) Add CLS token
        # cls_token (1,1,embed_dim) → expand → (B,1,embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # (B,1,embed_dim) + (B,100,embed_dim) → (B,101,embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # (4) Add Positional Embedding
        # pos_embed (1,101,embed_dim) → broadcast → (B,101,embed_dim)
        x = x + self.pos_embed

        # (5) Transformer encoder expects (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)  # (101, B, embed_dim)
        x = self.transformer(x)  # (101, B, embed_dim)

        # (6) Use CLS token output: x[0] → (B, embed_dim)
        cls_out = x[0]

        # (7) Regression Head: (B, embed_dim) → (B, 1) → squeeze → (B,)
        out = self.head(cls_out).squeeze(-1)
        return out

