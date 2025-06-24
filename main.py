import torch
from model import SiameseTransformer

# --- Model Instantiation ---
model = SiameseTransformer(
    image_size=224,
    patch_size=16,
    in_channels=1,
    cnn_out_channels=64,
    embed_dim=256,
    depth=4,
    heads=4,
    mlp_dim=1024
)

print("Model Architecture:")
print(model)
print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- Test with Dummy Data ---
dummy_img1 = torch.randn(4, 1, 224, 224)
dummy_img2 = torch.randn(4, 1, 224, 224)

embedding1, embedding2 = model(dummy_img1, dummy_img2)

print("\n--- Forward Pass Test ---")
print(f"Shape of Dummy Input 1: {dummy_img1.shape}")
print(f"Shape of Dummy Input 2: {dummy_img2.shape}")
print(f"Shape of Output Embedding 1: {embedding1.shape}")
print(f"Shape of Output Embedding 2: {embedding2.shape}")

# --- Triplet Loss Example ---
anchor_embedding = embedding1
positive_embedding = embedding2
negative_embedding = model.backbone(torch.randn(4, 1, 224, 224))

triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)

print(f"\nCalculated Triplet Loss (example): {loss.item()}")