from turtle import forward
import torch.nn as nn
import torch


class Space2Depth(nn.Module):
    """
    ref: https://github.com/huzi96/Coarse2Fine-PyTorch/blob/master/networks.py
    """

    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r**2)
        out_h = h // r
        out_w = w // r
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


class Depth2Space(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r**2)
        out_h = h * r
        out_w = w * r
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


def Demultiplexer(x):
    """
    See Supplementary Material: Figure 2.
    This operation can also implemented by slicing.
    """
    x_prime = Space2Depth(r=2)(x)
    
    _, C, _, _ = x_prime.shape
    anchor_index = tuple(range(C // 4, C * 3 // 4))
    non_anchor_index = tuple(range(0, C // 4)) + tuple(range(C * 3 // 4, C))
    
    anchor = x_prime[:, anchor_index, :, :]
    non_anchor = x_prime[:, non_anchor_index, :, :]

    return anchor, non_anchor

def Multiplexer(anchor, non_anchor):
    """
    The inverse opperation of Demultiplexer.
    This operation can also implemented by slicing.
    """
    _, C, _, _ = non_anchor.shape
    x_prime = torch.cat((non_anchor[:, : C//2, :, :], anchor, non_anchor[:, C//2:, :, :]), dim=1)
    return Depth2Space(r=2)(x_prime)


if __name__ == '__main__':
    x = torch.zeros(1, 1, 6, 6)
    x[0, 0, 0, 0] = 0
    x[0, 0, 0, 1] = 1
    x[0, 0, 1, 0] = 2
    x[0, 0, 1, 1] = 3
    print(x)

    anchor, non_anchor = Demultiplexer(x)
    print(anchor)
    print(non_anchor)

    x = Multiplexer(anchor, non_anchor)
    print(x)
