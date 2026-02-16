import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. DATA GENERATION: Synthetic 2D Manifold in D-dimensional Space
def generate_spiral_data(n_samples=5000, D=512, noise_std=0.1):
    # Create 2D Spiral (d=2)
    theta = torch.linspace(0, 4 * np.pi, n_samples)
    r = torch.linspace(0.1, 1, n_samples)
    x2d = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    
    # Embed into D-dimensional space via random orthogonal projection 
    # We create a fixed projection matrix P (D x 2)
    P = torch.randn(D, 2)
    Q, _ = torch.linalg.qr(P) # Ensure column-orthogonality [cite: 180]
    
    # Observed high-dim clean data
    x_clean_high = x2d @ Q.T 
    
    # Add noise for training samples
    noise = torch.randn_like(x_clean_high) * noise_std
    x_noisy = x_clean_high + noise
    
    return x_noisy, x_clean_high, Q

# 2. MODEL DEFINITIONS
class BaselineJiTMLP(nn.Module):
    """Reproduces the Paper's Linear Bottleneck[cite: 323, 329]."""
    def __init__(self, D=512, d_bottleneck=16, hidden_dim=256):
        super().__init__()
        # Linear Bottleneck Embedding: D -> d' -> hidden [cite: 330]
        self.bottleneck = nn.Sequential(
            nn.Linear(D, d_bottleneck, bias=False),
            nn.Linear(d_bottleneck, hidden_dim, bias=False)
        )
        # 5-layer ReLU MLP as described in the paper [cite: 170, 182]
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.predict = nn.Linear(hidden_dim, D) # x-prediction [cite: 191]

    def forward(self, z):
        h = self.bottleneck(z)
        h = self.trunk(h)
        return self.predict(h)

class ManifoldAwareMLP(nn.Module):
    """Experimental Idea: Non-linear bottleneck to capture manifold geometry."""
    def __init__(self, D=512, d_bottleneck=16, hidden_dim=256):
        super().__init__()
        # Non-linear Bottleneck (Linear -> ReLU -> Linear)
        self.bottleneck = nn.Sequential(
            nn.Linear(D, d_bottleneck),
            nn.ReLU(),
            nn.Linear(d_bottleneck, hidden_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.predict = nn.Linear(hidden_dim, D)

    def forward(self, z):
        h = self.bottleneck(z)
        h = self.trunk(h)
        return self.predict(h)

# 3. TRAINING LOOP
def train_model(model, noisy_data, clean_targets, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # Simple regression loss for x-prediction [cite: 161, 709]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(noisy_data)
        loss = criterion(preds, clean_targets)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

# Execution
D_dim = 512
d_bot = 16 # Paper found even 16-d works for 512-d input [cite: 333]
x_noisy, x_clean, Q = generate_spiral_data(D=D_dim)

print("Training Baseline (Linear Bottleneck)...")
baseline = train_model(BaselineJiTMLP(D=D_dim, d_bottleneck=d_bot), x_noisy, x_clean)

print("\nTraining Experimental (Manifold-Aware Bottleneck)...")
experimental = train_model(ManifoldAwareMLP(D=D_dim, d_bottleneck=d_bot), x_noisy, x_clean)

# 4. VISUALIZATION: Project back to 2D using Q [cite: 171, 182]
with torch.no_grad():
    base_preds = baseline(x_noisy) @ Q
    exp_preds = experimental(x_noisy) @ Q
    gt_2d = x_clean @ Q

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=5, alpha=0.5)
plt.title("Ground Truth (2D Manifold)")

plt.subplot(1, 3, 2)
plt.scatter(base_preds[:, 0], base_preds[:, 1], s=5, color='orange')
plt.title(f"Baseline (Linear {d_bot}-d Bottleneck)")

plt.subplot(1, 3, 3)
plt.scatter(exp_preds[:, 0], exp_preds[:, 1], s=5, color='green')
plt.title(f"Experimental (Non-linear {d_bot}-d Bottleneck)")
plt.show()
