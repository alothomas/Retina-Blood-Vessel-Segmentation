import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# Loss functions and metric functions

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute the Dice coefficient.
    
    Parameters:
        pred (torch.Tensor): Predicted mask, binary tensor [Height, Width].
        target (torch.Tensor): True mask, binary tensor [Height, Width].
        smooth (float): Smoothing term to avoid division by zero.
    
    Returns:
        dice_score (torch.Tensor): Computed Dice coefficient.
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice_score





class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs).squeeze()  
       
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DicePowerLoss(torch.nn.Module):
    def __init__(self, p_value=2.0, smooth=10):
        super(DicePowerLoss, self).__init__()
        self.p_value = p_value
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred).view(-1)
        y_true = y_true.view(-1)

        numerator = torch.sum(2.0 * (y_true * y_pred))

        y_true_pow = torch.pow(y_true, self.p_value)
        y_pred_pow = torch.pow(y_pred, self.p_value)
        denominator = torch.sum(y_true_pow) + torch.sum(y_pred_pow)

        dice_loss = (1 - ((numerator + self.smooth) / (denominator + self.smooth)))

        return dice_loss

def lovasz_hinge(logits, labels):
    """
    Binary Lovasz hinge loss
    logits: [N, H, W] Variable, logits at each pixel (before sigmoid)
    labels: [N, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    logits = logits.view(-1)
    labels = labels.view(-1)
    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    gt_sorted: Ground truth labels sorted by decreasing loss.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class BinaryLovaszHingeLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryLovaszHingeLoss, self).__init__()

    def forward(self, logits, labels):
        """
        logits: [N, 1, H, W] Variable, logits at each pixel (before sigmoid)
        labels: [N, H, W] Tensor, binary ground truth masks (0 or 1)
        """
        logits = logits.squeeze(1)  # Remove channel dim to match labels shape
        return lovasz_hinge(logits, labels)



class JaccardLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):

        y_pred = torch.sigmoid(y_pred).squeeze()  

        # Flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection 
        
        Jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - Jaccard


class JaccardPowerLoss(torch.nn.Module):
    def __init__(self, p_value=2.0, smooth=10):
        super(JaccardPowerLoss, self).__init__()
        self.p_value = p_value
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred).view(-1)
        y_true = y_true.view(-1)

        intersection = torch.sum(y_true * y_pred)

        term_true = torch.sum(torch.pow(y_true, self.p_value))
        term_pred = torch.sum(torch.pow(y_pred, self.p_value))

        union = term_true + term_pred - intersection

        Jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - Jaccard



############################################################################################################

def show_images_and_masks(dataset, num_imgs=3):
    fig, axs = plt.subplots(num_imgs, 2, figsize=(10, num_imgs * 5))
    
    for i in range(num_imgs):
        idx = np.random.randint(0, len(dataset))  # Randomly select an index
        image, mask = dataset[idx]
        
        # The image tensor might be normalized. Since we're skipping denormalization,
        # the image could appear with altered contrast/brightness.
        image = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC for plotting
        
        mask = mask.squeeze().numpy()  # Remove channel dim for mask (C=1)
        
        # Plot image
        axs[i, 0].imshow(image, cmap='gray')
        axs[i, 0].set_title(f"Image {idx}")
        axs[i, 0].axis('off')
        
        # Plot mask
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title(f"Mask {idx}")
        axs[i, 1].axis('off')
    
    plt.show()