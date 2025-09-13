import torch

checkpoint_fashion_mnist = torch.load("checkpoints/cnn_fashion_mnist_epoch.pth")
checkpoint_fmnist = torch.load("checkpoints/cnn_fmnist_checkpoint.pth")  

print(checkpoint_fashion_mnist.keys())