import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.80,
        gamma=2.0,
        smooth=1e-5,
        dice_weight=1.0,
        focal_weight=0.5,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        """
        logits : (B, 1, H, W)

        targets : (B, H, W) or (B, 1, H, W)
            Binary mask (0=background, 1=foreground).
        """

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # FOCAL LOSS
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # Correct class probabilities
        pt = torch.exp(-bce)

        # Foreground -> alpha
        # Background -> 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        focal_loss = focal_loss.mean()

        # DICE LOSS
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return (
            self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
        )
