import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the hyperspherical vMF distribution
class vMF(nn.Module):
    def __init__(self, dim, kappa):
        super(vMF, self).__init__()
        self.dim = dim
        self.kappa = kappa

    def sample(self, num_samples):
        # Generate hyperspherical samples using the vMF distribution
        samples = torch.randn(num_samples, self.dim).to(self.kappa.device)
        samples = F.normalize(samples, dim=1)
        return samples

    def log_prob(self, x):
        # Log-probability of the vMF distribution
        dim = self.dim
        kappa = self.kappa
        return kappa * (x[:, 0]) - (dim / 2) * torch.log(2 * torch.pi) - torch.log(kappa)

# Encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_kappa = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        kappa = F.softplus(self.fc_kappa(x))
        return mean, kappa

# Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(z))

# Hyperspherical VAE
class HypersphericalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(HypersphericalVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.vmf = vMF(latent_dim, torch.tensor(1.0))

    def forward(self, x):
        mean, kappa = self.encoder(x)
        z = self.vmf.sample(mean.shape[0])
        recon_x = self.decoder(z)
        return recon_x, mean, kappa, z

    def loss_function(self, recon_x, x, mean, kappa):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + kappa - mean.pow(2) - kappa.exp())
        return recon_loss + kl_div

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(mnist, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and device
input_dim = 28 * 28
latent_dim = 10
vae = HypersphericalVAE(input_dim, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# Training loop
epochs = 10
vae.train()
for epoch in range(epochs):
    train_loss = 0
    for x, _ in loader:
        x = x.view(-1, 28 * 28).to(device)
        optimizer.zero_grad()
        recon_x, mean, kappa, z = vae(x)
        loss = vae.loss_function(recon_x, x, mean, kappa)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(loader.dataset):.4f}")

print("Training complete.")

# Visualize reconstructed images
vae.eval()
dataiter = iter(loader)
images, labels = next(dataiter)
images = images.view(-1, 28 * 28).to(device)
recon_images, _, _, _ = vae(images)

# Plot original and reconstructed images
original = images.view(-1, 28, 28).cpu().detach()
reconstructed = recon_images.view(-1, 28, 28).cpu().detach()

fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for i in range(10):
    # Original images
    axes[0, i].imshow(original[i], cmap='gray')
    axes[0, i].axis('off')

    # Reconstructed images
    axes[1, i].imshow(reconstructed[i], cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=12)
axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
plt.show()
