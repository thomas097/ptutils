import torch
from torch import nn

class SepConv2d(nn.Module):
    """Implements a depthwise separable 2D convolution layer."""
    
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: int = 0, 
            dilation: int = 1, 
            use_bias: bool = True
            ) -> None:
        """
        Initializes a depthwise separable convolution layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        """
        super(SepConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=use_bias
        )
        
        self.pointwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            bias=use_bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after depthwise and pointwise convolutions.
        """
        h = self.depthwise(x)
        return self.pointwise(h)