import torch
import torch.optim as optim
import torch.nn.functional as F

class AnomalyDetectionAgent:
    def __init__(self, model, model_seg=None, learning_rate=0.001, reward_factor=0.1):
        self.model = model
        self.model_seg = model_seg
        if model_seg is not None:
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.model_seg.parameters()), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.reward_factor = reward_factor
        self.positive_reward = 1.0
        self.negative_reward = -1.0

    def adapt(self, data, target, prediction, pixel_mask=None):
        """
        Enhanced adaptation with pixel-level focus
        
        :param data: Input data (augmented image)
        :param target: Ground truth target (reconstructed image or mask)
        :param prediction: Model's prediction
        :param pixel_mask: Optional pixel-level anomaly mask
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(prediction, target, reduction='mean')
        
        # Add pixel-level focus if mask is provided
        if pixel_mask is not None:
            # Weight loss higher for anomalous regions
            pixel_weight = torch.ones_like(pixel_mask)
            pixel_weight[pixel_mask > 0.5] = 2.0  # Double weight for anomalous pixels
            pixel_loss = F.mse_loss(prediction * pixel_weight, target * pixel_weight, reduction='mean')
            total_loss = recon_loss + pixel_loss
        else:
            total_loss = recon_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def get_reward(self, target, prediction, pixel_mask=None):
        """
        Enhanced reward calculation with pixel-level consideration
        """
        with torch.no_grad():
            # Base reconstruction reward
            recon_reward = -F.mse_loss(prediction, target, reduction='mean')
            
            # Add pixel-level reward if mask is provided
            if pixel_mask is not None:
                # Calculate reward based on pixel-level accuracy
                pixel_pred = (prediction - target).abs() > 0.1
                pixel_reward = F.binary_cross_entropy_with_logits(
                    pixel_pred.float(), 
                    pixel_mask.float(), 
                    reduction='mean'
                )
                combined_reward = (recon_reward - pixel_reward) * self.reward_factor
            else:
                combined_reward = recon_reward * self.reward_factor
                
            return combined_reward

    def calculate_reward(self, l2_loss, ssim_loss, segment_loss):
        """
        Enhanced combined reward calculation
        """
        # Weight segment loss higher to focus more on pixel-level accuracy
        combined_loss = l2_loss + ssim_loss + 2.0 * segment_loss
        threshold = 0.4  # Lowered threshold for more sensitive reward
        reward = self.positive_reward if combined_loss < threshold else self.negative_reward
        return reward * self.reward_factor

