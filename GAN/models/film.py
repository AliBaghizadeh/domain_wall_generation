# Import Libraries
import torch
import torch.nn as nn

# Import Finished

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer for Conditional Normalization.

    This layer applies feature-wise affine transformations to an input tensor `x` 
    conditioned on discrete class labels. For each class, the layer learns 
    per-channel scaling (`gamma`) and bias (`beta`) parameters that modulate 
    the feature map activations:

        output = gamma[label] * x + beta[label]

    This is commonly used for class conditioning in image-to-image models such as
    conditional GANs and U-Nets.

    Args:
        num_features (int): Number of feature channels in the input tensor.
        n_classes (int): Number of discrete classes for conditioning.

    Inputs:
        x (torch.Tensor): Input feature map of shape (batch_size, num_features, H, W).
        labels (torch.LongTensor): Class labels of shape (batch_size,), containing
            integer values in the range [0, n_classes-1].

    Returns:
        torch.Tensor: Output tensor after FiLM modulation, same shape as input `x`.

    Raises:
        AssertionError: If a label value is outside the allowed range [0, n_classes-1].

    Example:
        film = FiLM(num_features=128, n_classes=6)
        out = film(x, labels)  # x: [B, 128, H, W], labels: [B]
    """
    def __init__(self, num_features, n_classes):
        super().__init__()
        self.gamma = nn.Embedding(n_classes, num_features)
        self.beta = nn.Embedding(n_classes, num_features)

    def forward(self, x, labels):
        # 🛡️ Runtime safety check
        max_index = self.gamma.num_embeddings - 1
        assert torch.max(labels).item() <= max_index, \
            f"[FiLM] Label index {torch.max(labels).item()} out of range (max allowed: {max_index})"

        gamma = self.gamma(labels).unsqueeze(2).unsqueeze(3)
        beta = self.beta(labels).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta
    

class LatentFiLM(nn.Module):
    """
       It modifies feature maps in a neural network using two sets of parameters.
       Inject an 8D noise vector z_wall ~ N(0, 1) into decoder layers via FiLM:
       Inject into UNetUp alongside label-based FiLM.
       """
    
    def __init__(self, z_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(z_dim, num_features)
        self.beta = nn.Linear(z_dim, num_features)

    def forward(self, x, z):
        gamma = self.gamma(z).unsqueeze(2).unsqueeze(3)
        beta = self.beta(z).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta
