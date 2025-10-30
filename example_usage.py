"""Example usage script for OPTICUS."""
import torch
from models import ViT50_3block
from utils import set_seed

# Set seed for reproducibility
set_seed(14000)

# Create a simple model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ViT50_3block(
    img_size=500,
    patch_size=50,
    embed_dim=256,
    depth=3,
    num_heads=8,
    mlp_dim=1024,
    num_classes=1
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel created successfully!")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randn(2, 1, 500, 500).to(device)
with torch.no_grad():
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

print("\n✓ All tests passed!")

