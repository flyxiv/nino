import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2d(nn.Module):
    def __init__(self):
        super(DynamicConv2d, self).__init__()
        
    def forward(self, x, weight, bias=None):
        """Apply dynamic convolution to the input tensor

        Args:
            x: input tensor (B, C_in, H, W)
            weight: dynamically generated convolution weights (B, C_out, C_in, kH, kW)
            bias: dynamically generated bias (B, C_out) or None
        
        Returns:
            output: convolution result (B, C_out, H_out, W_out)
        """
        batch_size = x.size(0)
        output = []
        
        for i in range(batch_size):
            if bias is not None:
                out = F.conv2d(x[i:i+1], weight[i], bias=bias[i])
            else:
                out = F.conv2d(x[i:i+1], weight[i])
            output.append(out)
        
        return torch.cat(output, dim=0)
