import torch


def dice_loss(seg_pred, mask, weight=None):
    """Note:
         If pred and mask (label) has multiple channels (denoting different semantics),
         the final loss will treat them additively, namely final loss won't be divided by n_channels.
    """
    batch_size, num_seg_classes = seg_pred.shape[:2]

    smooth = 1.0
    weight_dice = weight or [1.0] * num_seg_classes

    intersection = seg_pred.view(batch_size * num_seg_classes, -1) * mask.view(batch_size * num_seg_classes, -1)
    numerator = 2.0 * intersection.sum(1) + smooth
    denominator = (seg_pred ** 2 + mask ** 2).view(batch_size * num_seg_classes, -1).sum(1) + smooth
    # _loss = num_seg_classes - torch.sum(numerator / denominator) / batch_size
    _loss = torch.sum(
        (1.0 - torch.mean((numerator / denominator).view(batch_size, -1), 0))
        * torch.tensor(weight_dice, device=seg_pred.device)
    )

    return _loss
