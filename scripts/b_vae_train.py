# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import STL10, MNIST
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from typing import List, TypeVar
from tqdm import tqdm  # for progress bar
import numpy as np

Tensor = TypeVar("torch.tensor")


from ssl_methods.beta_vae.components import ResNet18Encoder, ResNet18Decoder
from ssl_methods.beta_vae.b_vae import BetaVae

# %% [markdown]
# ## POBRANIE DATASETOW
#


# %%
def get_dataset(dataset_name="stl10"):
    if dataset_name == "stl10":
        transform = transforms.Compose(
            [
                transforms.Resize((96, 96)),  # Resize to desired size
                transforms.ToTensor(),
            ]
        )
        train_dataset = STL10(
            root="./data", split="unlabeled", download=True, transform=transform
        )
        test_dataset = STL10(
            root="./data", split="test", download=True, transform=transform
        )
    elif dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        train_dataset = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Dataset not supported: choose 'stl10' or 'mnist'")

    return train_dataset, test_dataset


## MNIST was used to check quality of representation with easier dataset
dataset_name = "stl10"  # Change to 'mnist' to use the MNIST dataset
train_dataset, _ = get_dataset(dataset_name)


def get_subset_indices(dataset, percentage=0.01):
    num_samples = len(dataset)
    subset_size = int(num_samples * percentage)
    indices = np.random.choice(num_samples, subset_size, replace=False)
    return indices


subset_indices = get_subset_indices(train_dataset, percentage=0.25)
subset_dataset = Subset(train_dataset, subset_indices)

# Split the subset dataset into training and validation sets
train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])


# %% [markdown]
# ## TEST DLA WYBRANYCH HIPERPARAMETROW
#

# %%
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Training the VAE
num_epochs = 1

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )

else:
    device = torch.device("mps")

# Initialize the BetaVAE model
model = BetaVae(beta=2, in_channels=3, latent_dim=128).to(
    device
)  # Adjust latent_dim as needed

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
val_losses = []

early_stopping_patience = 10
best_val_loss = float("inf")
early_stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(
            device, dtype=torch.float32
        )  # Ensure data is float32 and on the correct device
        optimizer.zero_grad()

        recons, input, mu, log_var = model(data)
        loss_dict = model.loss_function(
            recons, input, mu, log_var, M_N=0.005
        )  # M_N is minibatch weight
        loss = loss_dict["loss"]

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_postfix({"Train Loss": train_loss / ((batch_idx + 1) * data.size(0))})

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device, dtype=torch.float32)
            recons, input, mu, log_var = model(data)
            loss_dict = model.loss_function(recons, input, mu, log_var, M_N=0.005)
            val_loss += loss_dict["loss"].item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}"
    )

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Plotting the training and test loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss Over Epochs")
plt.show()

# Display original and reconstructed images
with torch.no_grad():
    data_iter = iter(val_loader)
    images, _ = next(data_iter)
    images = images.to(device, dtype=torch.float32)
    recons, input, mu, log_var = model(images)

    # Convert images to CPU for matplotlib
    images = images.cpu()
    recons = recons.cpu()

    # Rescale images from [0, 1] to [0, 255]
    images = images * 255
    recons = recons * 255

    # Convert images to uint8 for proper display
    images = images.byte()
    recons = recons.byte()

    # Plot original images
    fig, axes = plt.subplots(
        nrows=2, ncols=8, sharex=True, sharey=True, figsize=(16, 4)
    )
    for imgs, row in zip([images, recons], axes):
        for img, ax in zip(imgs, row):
            ax.imshow(img.permute(1, 2, 0).numpy().astype("uint8"))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.suptitle("Top: Original images | Bottom: Reconstructed images")
    plt.show()

# %%
torch.save(model.state_dict(), "beta_vae_model2.pth")

# %% [markdown]
# ## WIZUALIZACJA REPREZENTACJI
#

# %%
import torch
import matplotlib.pyplot as plt


# Function to load a BetaVae model
def load_model(filepath, beta, in_channels, latent_dim):
    model = BetaVae(beta=beta, in_channels=in_channels, latent_dim=latent_dim)
    model.load_state_dict(torch.load(filepath))
    return model


# Initialize the device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    device = torch.device("cpu")
else:
    device = torch.device("mps")

# Model parameters
betas = [1, 2, 3, 4]
batch_size = 32
latent_dim = 64

# Load a batch of validation images
with torch.no_grad():
    data_iter = iter(val_loader)
    images, _ = next(data_iter)
    images = images.to(device, dtype=torch.float32)

# Plot original and reconstructed images for each model
fig, axes = plt.subplots(nrows=5, ncols=8, sharex=True, sharey=True, figsize=(16, 10))

# Plot original images in the first row
original_images = images.cpu() * 255
original_images = original_images.byte()

for img, ax in zip(original_images, axes[0]):
    ax.imshow(img.permute(1, 2, 0).numpy().astype("uint8"))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
axes[0, 0].set_ylabel("Original")

# Load and process each model
for i, beta in enumerate(betas):
    model_filename = f"beta_vae_beta_{beta}.pth"
    model = load_model(
        model_filename, beta=beta, in_channels=3, latent_dim=latent_dim
    ).to(device)
    model.eval()

    with torch.no_grad():
        recons, input, mu, log_var = model(images)

    recons = recons.cpu() * 255
    recons = recons.byte()

    for img, ax in zip(recons, axes[i + 1]):
        ax.imshow(img.permute(1, 2, 0).numpy().astype("uint8"))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    axes[i + 1, 0].set_ylabel(f"Beta {beta}")

fig.suptitle(
    "Top: Original images | Bottom: Reconstructed images for different Beta values"
)
plt.show()

# %% [markdown]
# ## GRID SEARCH PO PARAMETRACH
#
#

# %%
betas = [4, 3, 2, 1]
batch_sizes = [32]
latent_dims = [64]

for beta in betas:
    for batch_size in batch_sizes:
        for latent_dim in latent_dims:
            # Data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Training the VAE
            num_epochs = 50

            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                device = torch.device("cpu")
            else:
                device = torch.device("mps")

            # Initialize the BetaVAE model
            model = BetaVae(beta=beta, in_channels=3, latent_dim=latent_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = []
            val_losses = []

            early_stopping_patience = 5
            best_val_loss = float("inf")
            early_stopping_counter = 0

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
                for batch_idx, (data, _) in enumerate(pbar):
                    data = data.to(
                        device, dtype=torch.float32
                    )  # Ensure data is float32 and on the correct device
                    optimizer.zero_grad()

                    recons, input, mu, log_var = model(data)
                    loss_dict = model.loss_function(
                        recons, input, mu, log_var
                    )  # M_N is minibatch weight
                    loss = loss_dict["loss"]

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    pbar.set_postfix(
                        {"Train Loss": train_loss / ((batch_idx + 1) * data.size(0))}
                    )

                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)

                # Evaluate on validation set
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_idx, (data, _) in enumerate(val_loader):
                        data = data.to(device, dtype=torch.float32)
                        recons, input, mu, log_var = model(data)
                        loss_dict = model.loss_function(
                            recons, input, mu, log_var, M_N=0.005
                        )
                        val_loss += loss_dict["loss"].item()

                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Beta: {beta}, Batch Size: {batch_size}, Latent Dim: {latent_dim}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}"
                )

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    # Save the model when validation loss improves
                    model_filename = f"beta_vae_beta_{beta}.pth"
                    torch.save(model.state_dict(), model_filename)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print("Early stopping triggered")
                        break
