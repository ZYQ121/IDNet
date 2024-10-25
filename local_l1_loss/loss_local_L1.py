import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSimilarityLoss(nn.Module):
    def __init__(self, block_size, stride):
        super(LocalSimilarityLoss, self).__init__()
        self.block_size = block_size
        self.stride = stride

    def forward(self, x, y):
        x_blocks = x.unfold(2, self.block_size, self.stride).unfold(3, self.block_size, self.stride)
        y_blocks = y.unfold(2, self.block_size, self.stride).unfold(3, self.block_size, self.stride)

        x_blocks = x_blocks.reshape(-1, x.shape[1], self.block_size, self.block_size)
        y_blocks = y_blocks.reshape(-1, y.shape[1], self.block_size, self.block_size)

        loss = F.l1_loss(x_blocks, y_blocks, reduction='mean')

        return loss
