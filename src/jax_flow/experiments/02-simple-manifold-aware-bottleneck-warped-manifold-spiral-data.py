import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. NON-LINEAR DATA GENERATION
def generate_warped_manifold(n_samples=5000, D=512, noise_std=0.05):
    # Base 2D Spiral
    theta = torch.linspace(0, 4 * np.pi, n_samples)
    r = torch.linspace(0.1, 1, n_samples)
    x2d = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    
    # Non-Linear Embedding: Use a random fixed MLP to warp 2D into 512D
    # This simulates "natural" complex data manifolds
    with torch.no_grad():
        warper = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(), # Smooth non-linearity
            nn.Linear(128, D)
        )
        # Ensure the embedding is fixed
        for param in warper.parameters():
            param.requires_grad = False
            
    x_clean_high = warper(x2d)
    
    # Add noise
    noise = torch.randn_like(x_clean_high) * noise_std
    x_noisy = x_clean_high + noise
    
    return x_noisy, x_clean_high, x2d # Returning x2d for ground truth visualization

# 2. MODELS (Keeping your Non-Linear Bottleneck idea)
class BaselineJiTMLP(nn.Module):
    def __init__(self, D=512, d_bottleneck=16, hidden_dim=256):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(D, d_bottleneck, bias=False),
            nn.Linear(d_bottleneck, hidden_dim, bias=False)
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.predict = nn.Linear(hidden_dim, D)

    def forward(self, z):
        return self.predict(self.trunk(self.bottleneck(z)))

class ManifoldAwareMLP(nn.Module):
    def __init__(self, D=512, d_bottleneck=16, hidden_dim=256):
        super().__init__()
        # YOUR IDEA: Non-linear bottleneck
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
        return self.predict(self.trunk(self.bottleneck(z)))

# 3. TRAINING & VISUALIZATION
D_dim = 512
d_bot = 8 # Tight bottleneck to force representation learning [cite: 79]
x_noisy, x_clean, x2d_gt = generate_warped_manifold(D=D_dim)

def run_train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(500):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x_noisy), x_clean)
        loss.backward()
        optimizer.step()
    return model, loss

print("Training Models on Warped Manifold...")
base_model, base_loss = run_train(BaselineJiTMLP(D=D_dim, d_bottleneck=d_bot))
print(f"Base loss: {base_loss}")
exp_model, candidate_loss  = run_train(ManifoldAwareMLP(D=D_dim, d_bottleneck=d_bot))
print(f"Candidate loss: {candidate_loss}")

# For visualization, we use PCA to project 512D back to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
clean_2d = pca.fit_transform(x_clean.numpy())
base_2d = pca.transform(base_model(x_noisy).detach().numpy())
exp_2d = pca.transform(exp_model(x_noisy).detach().numpy())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(clean_2d[:, 0], clean_2d[:, 1], s=5, alpha=0.5); axes[0].set_title("Target Manifold (PCA)")
axes[1].scatter(base_2d[:, 0], base_2d[:, 1], s=5, color='orange'); axes[1].set_title("Linear Bottleneck Recovery")
axes[2].scatter(exp_2d[:, 0], exp_2d[:, 1], s=5, color='green'); axes[2].set_title("Non-Linear Bottleneck Recovery")
plt.show()
