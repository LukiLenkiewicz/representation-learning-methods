import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import seaborn as sns

def plot_loss(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    plt.figure(figsize=(15, 10))
    sns.plot(checkpoint["train_d_loss"], label="train_d_loss", color="red")
    sns.plot(checkpoint["train_g_loss"], label="train_g_loss", color="blue")
    sns.plot(checkpoint["val_d_loss"], label="val_d_loss", color="green")
    sns.plot(checkpoint["val_g_loss"], label="val_g_loss", color="yellow")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BiGAN Loss")
    plt.show()
    