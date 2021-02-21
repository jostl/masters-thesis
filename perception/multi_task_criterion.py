import torch.nn as nn


class MultiTaskCriterion(nn.Module):
    def __init__(self):
        # TODO undersøk bedre måter å regne ut multi-task loss på
        super(MultiTaskCriterion, self).__init__()
        self.rgb_loss = nn.MSELoss()
        #self.semantic_loss = self.dice_loss
        self.semantic_loss = nn.CrossEntropyLoss()
        # Todo finne bedre loss function for depth
        self.depth_loss = nn.MSELoss()

        self.rgb_weight = 0.1
        self.semantic_weight = 0.8
        self.depth_weight = 0.1

    def forward(self, predictions, targets):
        rgb_pred, semantic_pred, depth_pred = predictions
        rgb_target, semantic_target, depth_target = targets

        # Calculate RGB loss
        rgb_loss = self.rgb_loss(rgb_pred, rgb_target)

        semantic_target = semantic_target.argmax(dim=1)
        # Calculate semantic segmentation loss
        semantic_loss = self.semantic_loss(semantic_pred, semantic_target)
        depth_loss = self.depth_loss(depth_pred, depth_target)
        #print("rgb loss {}, depth loss {}, semantic loss {}".format(rgb_loss, depth_loss, semantic_loss))
        return self.rgb_weight * rgb_loss + self.semantic_weight * semantic_loss + self.depth_weight * depth_loss

    def dice_loss(self, semantic_pred, semantic_target, smooth=1.):
        # This function is copied from: https://github.com/usuyama/pytorch-unet/blob/master/loss.py
        semantic_pred = semantic_pred.contiguous()
        semantic_target = semantic_target.contiguous()
        intersection = (semantic_pred * semantic_target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (
                semantic_pred.sum(dim=2).sum(dim=2) + semantic_target.sum(dim=2).sum(dim=2) + smooth)))

        return loss.mean()
