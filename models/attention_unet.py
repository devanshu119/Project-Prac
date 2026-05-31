# file: models/attention_unet.py
"""
Attention U-Net with CBAM (Convolutional Block Attention Module).

This architecture extends the standard U-Net with attention gates at each
skip connection, allowing the model to focus on salient features and suppress
irrelevant activations. Particularly effective for SAR oil spill detection
where spill boundaries are subtle against ocean clutter.

Reference:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
    Woo et al., "CBAM: Convolutional Block Attention Module"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention sub-module of CBAM.
    
    Applies both average-pooling and max-pooling across spatial dimensions,
    passes through a shared MLP, and produces per-channel attention weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pool → (B, C)
        avg_pool = x.view(b, c, -1).mean(dim=2)
        # Global max pool → (B, C)
        max_pool = x.view(b, c, -1).max(dim=2)[0]

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        attn = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial Attention sub-module of CBAM.
    
    Concatenates channel-wise average and max features, then applies
    a 7×7 convolution to produce a spatial attention map.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Sequential application of Channel Attention followed by Spatial Attention.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate for skip connections.
    
    Learns to weight skip-connection features based on contextual
    information from the decoder path (gating signal).
    
    Args:
        F_g: Number of channels in the gating signal (from decoder).
        F_l: Number of channels in the skip connection (from encoder).
        F_int: Number of intermediate channels.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from the decoder (lower resolution).
            x: Skip connection from the encoder (higher resolution).
        
        Returns:
            Attention-weighted skip features.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align spatial dimensions (g may be smaller than x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConvBN(nn.Module):
    """(Conv2d → BN → ReLU → Dropout) × 2 with optional CBAM."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1, use_cbam: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net with CBAM attention at each level.

    Architecture:
        Encoder: 4 downsampling stages with CBAM-enhanced double convolutions
        Bottleneck: Deep feature extraction with CBAM
        Decoder: 4 upsampling stages with Attention Gates on skip connections
        Output: 1×1 convolution to n_classes

    Args:
        n_channels: Number of input channels (1 for grayscale SAR).
        n_classes: Number of output segmentation classes.
        base_filters: Number of filters in the first encoder stage (doubled at each level).
        dropout: Dropout probability for regularization.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 2,
                 base_filters: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        f = base_filters  # shorthand

        # ── Encoder ──
        self.enc1 = DoubleConvBN(n_channels, f, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvBN(f, f * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvBN(f * 2, f * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConvBN(f * 4, f * 8, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = DoubleConvBN(f * 8, f * 16, dropout=dropout)

        # ── Decoder with Attention Gates ──
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(F_g=f * 8, F_l=f * 8, F_int=f * 4)
        self.dec4 = DoubleConvBN(f * 16, f * 8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.dec3 = DoubleConvBN(f * 8, f * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=f * 2, F_l=f * 2, F_int=f)
        self.dec2 = DoubleConvBN(f * 4, f * 2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=f, F_l=f, F_int=f // 2)
        self.dec1 = DoubleConvBN(f * 2, f, dropout=dropout)

        # ── Output ──
        self.outc = nn.Conv2d(f, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)                        # (B, 64, H, W)
        e2 = self.enc2(self.pool1(e1))            # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))            # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool3(e3))            # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))       # (B, 1024, H/16, W/16)

        # Decoder path with attention-gated skip connections
        d4 = self.up4(b)                          # (B, 512, H/8, W/8)
        e4 = self._pad_to_match(e4, d4)
        e4_attn = self.attn4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([e4_attn, d4], dim=1))

        d3 = self.up3(d4)                         # (B, 256, H/4, W/4)
        e3 = self._pad_to_match(e3, d3)
        e3_attn = self.attn3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([e3_attn, d3], dim=1))

        d2 = self.up2(d3)                         # (B, 128, H/2, W/2)
        e2 = self._pad_to_match(e2, d2)
        e2_attn = self.attn2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([e2_attn, d2], dim=1))

        d1 = self.up1(d2)                         # (B, 64, H, W)
        e1 = self._pad_to_match(e1, d1)
        e1_attn = self.attn1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([e1_attn, d1], dim=1))

        return self.outc(d1)                      # (B, n_classes, H, W)

    @staticmethod
    def _pad_to_match(encoder_feat, decoder_feat):
        """Pad encoder feature map to match decoder spatial dimensions."""
        diff_h = decoder_feat.size(2) - encoder_feat.size(2)
        diff_w = decoder_feat.size(3) - encoder_feat.size(3)
        if diff_h != 0 or diff_w != 0:
            encoder_feat = F.pad(encoder_feat, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
            ])
        return encoder_feat
