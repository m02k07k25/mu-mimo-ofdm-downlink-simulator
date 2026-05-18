"""
DL 모델 3종.

[1] LSRefineNet (CE subnet — design3/4 매칭)
    입력 : 2·K  per subcarrier  (per-user diagonal He_est, real/imag)
    출력 : 2·K  refined He
    구조 : Linear(2K, 2K, bias=False), LMMSE init

[2] FC-SD (ComNet SD subnet — design3/4 매칭)
    입력 : 6 per (k, user) — (Re/Im y, Re/Im h, Re/Im x_zf)
    출력 : 64-class softmax (64-QAM 분류)
    구조 : Linear(6, 256)→ReLU→Linear(256, 256)→ReLU→Linear(256, 64)

[3] E2E-NN (MATLAB main.m 매칭)
    입력 : 4 per (k, user) — (Re/Im y, Re/Im h)
    출력 : 64-class softmax
    구조 : Linear(4, 512)→ReLU→Linear(512, 256)→ReLU→Linear(256, 64)
"""
import torch
import torch.nn as nn


class LSRefineNet(nn.Module):
    """K-user per-subcarrier diagonal channel refiner."""
    def __init__(self, num_users):
        super().__init__()
        d = num_users
        self.linear = nn.Linear(2 * d, 2 * d, bias=False)
        self.K = num_users

    def forward(self, x_ri):                            # (B, 2K)
        return self.linear(x_ri)


class FCSD(nn.Module):
    """ComNet SD subnet — per-(k, user) classifier."""
    def __init__(self, hidden=256, num_classes=64, dropout=0.0):
        super().__init__()
        in_dim = 6      # Re/Im of y, h, x_zf
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class E2ENet(nn.Module):
    """MATLAB main.m 의 단대단 DL 수신기 — FC 512-256-64."""
    def __init__(self, num_classes=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
