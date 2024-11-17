import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Convert target to long and squeeze extra dimension
        target = target.squeeze(1).long()
        
        # Compute focal loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # weight for reconstruction
        self.beta = beta    # weight for SSIM
        self.gamma = gamma  # weight for segmentation
        
        self.ssim = SSIM()
        self.focal = FocalLoss()
        
    def forward(self, pred, target, seg_pred=None, seg_target=None):
        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        
        # SSIM loss
        ssim_loss = self.ssim(pred, target)
        
        # Segmentation loss if provided
        if seg_pred is not None and seg_target is not None:
            seg_loss = self.focal(seg_pred, seg_target)
            return (self.alpha * recon_loss + 
                   self.beta * ssim_loss + 
                   self.gamma * seg_loss)
        
        return self.alpha * recon_loss + self.beta * ssim_loss

class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(EnhancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # Make sure target is proper shape
        target = target.squeeze(1)
        
        # Convert to binary classification
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        return focal_loss.mean()
    
    def calculate_edge_weight(self, target):
        # Calculate edge regions using gradients
        dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        dy = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Pad to original size
        dx = F.pad(dx, (0, 0, 1, 0))
        dy = F.pad(dy, (1, 0, 0, 0))
        
        # Edge weight
        edge = torch.sqrt(dx*dx + dy*dy)
        weight = 1.0 + edge * 2.0  # Increase weight near edges
        
        return weight