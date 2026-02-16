import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Check for device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. NON-LINEAR DATA GENERATION
def generate_warped_manifold(n_samples=5000, D=512, noise_std=0.05):
    # Base 2D Spiral (the underlying d=2 manifold)
    theta = torch.linspace(0, 4 * np.pi, n_samples)
    r = torch.linspace(0.1, 1, n_samples)
    x2d = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).to(device)
    
    # Non-Linear Embedding: A fixed random MLP to warp 2D into 512D
    with torch.no_grad():
        warper = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, D)
        ).to(device)
        for param in warper.parameters():
            param.requires_grad = False
            
    x_clean_high = warper(x2d)
    noise = torch.randn_like(x_clean_high) * noise_std
    x_noisy = x_clean_high + noise
    
    return x_noisy, x_clean_high

# 2. BASELINE MODEL: JiT with Linear Bottleneck [cite: 51, 323]
class BaselineJiT(nn.Module):
    def __init__(self, D=512, d_bot=16, hidden_dim=256):
        super().__init__()
        # The paper advocates for a linear bottleneck to encourage low-dimensional representations [cite: 328, 336]
        self.bottleneck = nn.Sequential(
            nn.Linear(D, d_bot, bias=False),
            nn.Linear(d_bot, hidden_dim, bias=False)
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.predict = nn.Linear(hidden_dim, D) # x-prediction [cite: 48, 211]

    def forward(self, x):
        return self.predict(self.trunk(self.bottleneck(x)))

# 3. EXPERIMENTAL MODEL: Soft Rank Selector
class SoftRankJiT(nn.Module):
    def __init__(self, D=512, d_bot=16, hidden_dim=256):
        super().__init__()
        self.compress = nn.Linear(D, d_bot, bias=False)
        # Learnable gating parameter to autonomously select manifold rank
        self.soft_gate = nn.Parameter(torch.ones(d_bot)) 
        self.expand = nn.Linear(d_bot, hidden_dim, bias=False)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.predict = nn.Linear(hidden_dim, D)

    def forward(self, x):
        h = self.compress(x)
        # Apply gating to select dimensions
        h = h * torch.sigmoid(self.soft_gate)
        return self.predict(self.trunk(self.expand(h)))

# 4. FULL EXPERIMENTAL SETUP
D_dim, d_bot = 512, 16
x_noisy, x_clean = generate_warped_manifold(D=D_dim)

def train_and_eval(model, is_experimental=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # x-loss [cite: 165]
    
    for epoch in range(1500):
        optimizer.zero_grad()
        preds = model(x_noisy)
        loss = criterion(preds, x_clean)
        
        if is_experimental:
            # Add L1 sparsity penalty to the soft_gate to prune noise dimensions
            sparsity = 1e-4 * torch.norm(torch.sigmoid(model.soft_gate), 1)
            loss += sparsity
            
        loss.backward()
        optimizer.step()
    return model, loss

print("Training Baseline (Linear Bottleneck)...")
baseline_model, loss = train_and_eval(BaselineJiT(D=D_dim, d_bot=d_bot).to(device))
print(f"Baseline loss {loss}")

print("Training Experimental (Soft Rank Selector)...")
experimental_model, loss = train_and_eval(SoftRankJiT(D=D_dim, d_bot=d_bot).to(device), is_experimental=True)
print(f"Experimental loss {loss}")

# 5. VISUALIZATION
with torch.no_grad():
    pca = PCA(n_components=2)
    clean_np = x_clean.cpu().numpy()
    target_2d = pca.fit_transform(clean_np)
    
    base_preds = baseline_model(x_noisy).cpu().numpy()
    exp_preds = experimental_model(x_noisy).cpu().numpy()
    
    base_2d = pca.transform(base_preds)
    exp_2d = pca.transform(exp_preds)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(target_2d[:, 0], target_2d[:, 1], s=2, alpha=0.5)
axes[0].set_title("Ground Truth Manifold (PCA)")

axes[1].scatter(base_2d[:, 0], base_2d[:, 1], s=2, color='orange')
axes[1].set_title("Baseline (Linear Bottleneck)")

axes[2].scatter(exp_2d[:, 0], exp_2d[:, 1], s=2, color='green')
axes[2].set_title("Experimental (Soft Rank Selector)")

plt.show()

# Print active dimensions for Experimental Model
with torch.no_grad():
    gates = torch.sigmoid(experimental_model.soft_gate)
    active = (gates > 0.5).sum().item()
    print(f"Experimental Model Active Dimensions: {active} / {d_bot}")
