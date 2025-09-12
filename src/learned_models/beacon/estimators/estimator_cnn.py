import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Small CNN for regressing [x, y, vx, vy] from a 3xHxW image.
    Conv -> ReLU -> Conv -> ReLU -> Flatten -> MLP (64->32->out_dim).
    No pooling, no BN, no residuals (LiRPA-friendly).
    """
    def __init__(self, input_channels=3, out_dim=4, hidden1=64, hidden2=32, H=224, W=224):
        super().__init__()
        # Two lightweight conv layers; strides chosen to reduce size
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0)

        # Dynamically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, H, W)
            dummy_out = self._forward_convs(dummy)
            flatten_size = dummy_out.view(1, -1).size(1)

        # MLP head
        self.fc1 = nn.Linear(flatten_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)

    def _forward_convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self._forward_convs(z)
        x = torch.flatten(x, 1)  # (B, flatten_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # linear output
        return x


# class CNN(nn.Module):
#     """
#     Small CNN for regressing [cte, heading_error] from a 3xHxW image.
#     Conv -> ReLU -> Conv -> ReLU -> Flatten -> MLP.
#     No pooling, no BN, no residuals (LiRPA-friendly).
#     """
#     def __init__(self, input_channels=3, out_dim=2, hidden1=64, hidden2=32, H=224, W=224):
#         super().__init__()
#         # Two lightweight conv layers
#         self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=4, padding=0)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0)

#         # Find flatten size with a dummy forward pass
#         with torch.no_grad():
#             dummy = torch.zeros(1, input_channels, H, W)
#             dummy_out = self._forward_convs(dummy)
#             flatten_size = dummy_out.numel()

#         # MLP head
#         self.fc1 = nn.Linear(flatten_size, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, out_dim)

#     def _forward_convs(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         return x

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = self._forward_convs(z)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)  # linear output
#         return x

# class CNN(nn.Module):
#     """
#     Small CNN for regressing [x, y, vx, vy] from a 3x224x224 image.
#     Ops: Conv -> ReLU -> Conv -> ReLU -> Flatten -> MLP (64->32->4).
#     No pooling, no BN, no residuals (LiRPA-friendly).
#     """
#     def __init__(self, input_channels=3, H=224, W=224):
#         super(CNN, self).__init__()
#         # Two lightweight conv layers; strides chosen to keep FC size small.
#         self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=4, padding=0)  # -> 56x56
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0)              # -> 14x14

#         # Compute flatten size analytically for the given H, W
#         def out_dim(n, k, s, p, d=1):
#             return (n + 2*p - d*(k - 1) - 1) // s + 1

#         h1 = out_dim(H, 5, 4, 2)
#         w1 = out_dim(W, 5, 4, 2)
#         h2 = out_dim(h1, 3, 4, 1)
#         w2 = out_dim(w1, 3, 4, 1)
#         flatten_size = 32 * h2 * w2  # 32 * 14 * 14 = 6272 for 224x224

#         # MLP head
#         self.fc1 = nn.Linear(flatten_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)  # [cross_track_error, heading_error]

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.conv1(z))
#         x = F.relu(self.conv2(x))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)  # no activation on output
#         return x



# class CNN(nn.Module):
#     def __init__(self, in_channels=3, out_dim=4, hidden=128):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # (B,16,H/2,W/2)
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # (B,32,H/4,W/4)
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # (B,64,H/8,W/8)
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4)),  # fixed-size regardless of input H,W
#         )

#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, hidden), nn.ReLU(),
#             nn.Linear(hidden, out_dim),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.head(x)
#         return x
