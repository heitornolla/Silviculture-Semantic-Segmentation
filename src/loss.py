import torch.nn as nn
import torch.nn.functional as F

# Semantic segmentation of silviculture is a naturally imbalanced problem
# I fear merely undersampling may lead to poor results, given that there is valuable
#  information in background texture (e.g. the model should also learn to predict nothing)

# Famous losses for imbalanced problems are Focal, Dice and Tversky
# The main idea here is that Dice will be the global optimizer and Focal the local one
# If the model shows too many false negatives (fails to predict silviculture) one may up dice_weight
# If it leaks (predicts silviculture too much), up the focal_weight and gamma factors

# If the network still suffers from too many false positives, change Dice for Tversky loss
# It allows for separate balancing of FPs and FNs
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-5, dice_weight=1.0, focal_weight=1.0):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        # inputs  (Batch, 2, H, W)
        # targets (Batch, H, W)
        
        probs = F.softmax(inputs, dim=1)
        
        p = probs[:, 1, :, :]
        targets_float = targets.float()
        
        # FOCAL LOSS
        bce = F.binary_cross_entropy(p, targets_float, reduction='none')
        
        # pt = prob of true
        p_t = p * targets_float + (1 - p) * (1 - targets_float)
        
        # Focal: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        focal_loss = focal_loss.mean()
        
        # DICE LOSS
        intersection = (p * targets_float).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + targets_float.sum(dim=(1, 2))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        total_loss = (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)
        
        return total_loss
