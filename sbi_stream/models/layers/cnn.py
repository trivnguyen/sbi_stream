"""2D CNN feature extractor layers: plain CNN and ResNet-style."""

from typing import List, Union, Optional

import torch
import torch.nn as nn


class CNN(nn.Module):
    """2D CNN feature extractor.

    Each block applies: Conv2d → [BatchNorm2d] → act → [Dropout2d] → [MaxPool2d].
    A final AdaptiveAvgPool2d + Flatten produces a fixed-size feature vector
    regardless of input spatial dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : list of int
        Output channels for each convolutional block.
    kernel_size : int or list of int
        Convolution kernel size per block. If a single int, the same size
        is applied to all blocks. Default: 3.
    stride : int or list of int
        Convolution stride per block. Default: 1.
    downsample : bool or list of bool
        Whether to apply 2×2 MaxPool2d after each block for spatial
        downsampling. Default: False.
    batch_norm : bool
        Add BatchNorm2d after each convolution. Default: True.
    dropout : float
        Channel-wise Dropout2d probability applied after each block. Default: 0.0.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    pooling_output_size : int
        Spatial output size for the final AdaptiveAvgPool2d. The default of 1
        gives global average pooling and an output feature size of
        ``channels[-1]``. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: Union[int, List[int]] = 3,
        stride: Union[int, List[int]] = 1,
        downsample: Union[bool, List[bool]] = False,
        batch_norm: bool = True,
        dropout: float = 0.0,
        act: nn.Module = None,
        pooling_output_size: int = 1,
    ):
        super().__init__()

        if act is None:
            act = nn.ReLU()

        n_blocks = len(channels)
        kernel_sizes = [kernel_size] * n_blocks if isinstance(kernel_size, int) else list(kernel_size)
        strides = [stride] * n_blocks if isinstance(stride, int) else list(stride)
        downsamples = [downsample] * n_blocks if isinstance(downsample, bool) else list(downsample)

        blocks = []
        in_ch = in_channels
        for out_ch, ks, st, do_pool in zip(channels, kernel_sizes, strides, downsamples):
            padding = ks // 2  # preserve spatial dims for odd kernels at stride=1
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=st, padding=padding))
            if batch_norm:
                blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(act)
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
            if do_pool:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pooling_output_size)
        self.flatten = nn.Flatten()
        self.output_size = channels[-1] * pooling_output_size ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(B, output_size)``.
        """
        x = self.conv_blocks(x)
        x = self.pool(x)
        return self.flatten(x)


# ---------------------------------------------------------------------------
# ResNet-style blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-activation residual block (He et al., 2016).

    Architecture: BN → act → Conv2d → BN → act → Conv2d + skip.
    A 1×1 projection conv is added automatically when ``in_channels`` !=
    ``out_channels`` or ``stride`` > 1.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Stride for the first convolution (use 2 to halve spatial dimensions).
        Default: 1.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        act: nn.Module = None,
    ):
        super().__init__()
        if act is None:
            act = nn.ReLU()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            act,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Projection shortcut to match dimensions when needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class ResNet(nn.Module):
    """Flexible ResNet feature extractor.

    Consists of an initial stem convolution followed by a configurable
    number of residual stages. Each stage contains one or more
    :class:`ResidualBlock` blocks, with the first block of each stage
    (after the first) halving spatial dimensions via ``stride=2``.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : list of int
        Number of output channels for each stage.
        ``len(channels)`` determines the number of stages.
    blocks_per_stage : int or list of int
        Number of residual blocks per stage. Default: 2.
    stem_channels : int
        Output channels of the initial 7×7 stem convolution. Default: 64.
    stem_stride : int
        Stride of the stem convolution. Use 2 to halve spatial dims early
        (as in the original ResNet for large images). Default: 1.
    downsample_stages : bool or list of bool
        Whether to apply stride=2 at the start of each stage (to halve
        spatial dimensions). Default: True for all stages except the first.
        If a single bool, applies to all stages.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    pooling_output_size : int
        Spatial output size for the final AdaptiveAvgPool2d. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        blocks_per_stage: Union[int, List[int]] = 2,
        stem_channels: int = 64,
        stem_stride: int = 1,
        downsample_stages: Union[bool, List[bool]] = None,
        act: nn.Module = None,
        pooling_output_size: int = 1,
    ):
        super().__init__()

        if act is None:
            act = nn.ReLU()

        n_stages = len(channels)

        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * n_stages

        # Default: downsample at every stage after the first
        if downsample_stages is None:
            downsample_stages = [False] + [True] * (n_stages - 1)
        elif isinstance(downsample_stages, bool):
            downsample_stages = [downsample_stages] * n_stages

        # Stem: single large conv to process the raw image
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7,
                      stride=stem_stride, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            act,
        )

        # Residual stages
        stages = []
        prev_ch = stem_channels
        for out_ch, n_blocks, do_ds in zip(channels, blocks_per_stage, downsample_stages):
            stage = []
            for b in range(n_blocks):
                stride = 2 if (b == 0 and do_ds) else 1
                stage.append(ResidualBlock(prev_ch, out_ch, stride=stride, act=act))
                prev_ch = out_ch
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(pooling_output_size)
        self.flatten = nn.Flatten()
        self.output_size = channels[-1] * pooling_output_size ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(B, output_size)``.
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.flatten(x)
