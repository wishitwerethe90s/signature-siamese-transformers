import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

class HybridBackbone(nn.Module):
    """
    The core feature extractor combining a CNN front-end with a Transformer back-end.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=1, cnn_out_channels=64, embed_dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        
        # 1. CNN Front-end for Local Feature Extraction : Simple structure
        # self.cnn_feature_extractor = nn.Sequential(
        #     nn.Conv2d(in_channels, cnn_out_channels, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(cnn_out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        # )

        # 1. Improved CNN front-end for local feature extraction TODO: Review
        self.cnn_feature_extractor = nn.Sequential(
            # Stage 1: Initial feature extraction
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 2: More detailed features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Stage 3: High-level features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Final projection to embedding dimension
            nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=patch_size)
        )

        # 2. Transformer Back-end for Global Context
        config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=mlp_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=in_channels,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.transformer_encoder = ViTModel(config).encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # TODO: review patch sizing; need to make sure there are no runtime errors.
        def calculate_patches(image_size, patch_size):
            after_cnn = image_size // 4  # Two stride-2 operations
            return (after_cnn // patch_size) ** 2

        num_patches = calculate_patches(image_size, patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))


        # ***TODO: Further refinements to architecture; Review***
        # Add stroke-aware attention
        self.stroke_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=4, 
            dropout=0.1
        )
        
        # Add feature refinement
        self.feature_refiner = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        # Pass through CNN front-end
        x = self.cnn_feature_extractor(x)  # Shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        
        # Flatten and prepare for Transformer
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Prepend the CLS token
        batch_size = x.shape[0]
        # Correctly expand the CLS token to match the batch size.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.position_embeddings
        
        # Pass through Transformer Encoder
        encoder_outputs = self.transformer_encoder(x)
        
        # The final embedding is the output of the CLS token
        embedding = encoder_outputs.last_hidden_state[:, 0]


        # ***TODO: Further refinements to architecture; Review***
        # Apply stroke-aware attention
        refined_features, _ = self.stroke_attention(
            encoder_outputs.last_hidden_state,
            encoder_outputs.last_hidden_state,
            encoder_outputs.last_hidden_state
        )
        
        # Final embedding with refinement
        cls_embedding = refined_features[:, 0]
        embedding = self.feature_refiner(cls_embedding)
        
        return embedding

class SiameseTransformer(nn.Module):
    """
    The complete Siamese network. It takes two images as input and returns their embeddings.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=1, cnn_out_channels=64, embed_dim=768, depth=6, heads=8, mlp_dim=3072):
        super().__init__()
        self.backbone = HybridBackbone(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            cnn_out_channels=cnn_out_channels,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, input1, input2):
        output1 = self.backbone(input1)
        output2 = self.backbone(input2)
        return output1, output2