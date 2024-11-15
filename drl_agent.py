import torch
import torch.optim as optim
import torch.nn.functional as F


# Reward-based adaptation for anomaly detection
class AnomalyDetectionAgent:
    def __init__(self, model, model_seg=None, learning_rate=0.001, reward_factor=0.1):
        """
        Initialize the Anomaly Detection Agent with both reconstruction and segmentation models.
        
        :param model: ReconstructiveSubNetwork model
        :param model_seg: DiscriminativeSubNetwork model (optional)
        :param learning_rate: Learning rate for adaptation
        :param reward_factor: Scaling factor for the reward
        """
        self.model = model
        self.model_seg = model_seg
        if model_seg is not None:
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.model_seg.parameters()), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.reward_factor = reward_factor
        self.positive_reward = 1.0
        self.negative_reward = -1.0

    def adapt(self, data, target, prediction):
        """
        Adapt the model to unknown anomalies based on reward signals.
        
        :param data: Input data (augmented image)
        :param target: Ground truth target (reconstructed image or mask)
        :param prediction: Model's prediction (reconstructed image or mask)
        """
        # Modify: Removed reward from loss calculation to avoid issues with scalar gradients
        loss = F.mse_loss(prediction, target, reduction='mean')  # Ensure the loss is a scalar

        self.optimizer.zero_grad()
        loss.backward()  # Removed retain_graph=True to prevent reusing the same graph

        self.optimizer.step()

    def get_reward(self, target, prediction):
        """
        Calculate the reward based on detection performance.
        
        :param target: Ground truth target
        :param prediction: Model's prediction
        :return: Calculated reward value
        """
        # Modify: Adjusted to use a more appropriate metric for continuous outputs
        with torch.no_grad():  # Detach tensors from the computational graph to prevent retaining it
            reward = -F.mse_loss(prediction, target, reduction='mean')  # Negative MSE as reward
        return reward * self.reward_factor

    def calculate_reward(self, l2_loss, ssim_loss, segment_loss):
        """
        Calculate a combined reward based on the reconstruction and segmentation losses.
        
        :param l2_loss: L2 reconstruction loss
        :param ssim_loss: SSIM reconstruction loss
        :param segment_loss: Segmentation loss
        :return: Combined reward value
        """
        combined_loss = l2_loss + ssim_loss + segment_loss
        reward = self.positive_reward if combined_loss < 0.5 else self.negative_reward  # Threshold can be tuned
        return reward * self.reward_factor

if __name__ == "__main__":
    # Example usage of the Anomaly Detection Agent for testing purposes
    import argparse
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for adaptation')
    parser.add_argument('--reward_factor', type=float, default=0.1, help='Scaling factor for reward')
    args = parser.parse_args()

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    agent = AnomalyDetectionAgent(model, model_seg, learning_rate=args.learning_rate, reward_factor=args.reward_factor)
    print("AnomalyDetectionAgent initialized successfully with learning rate:", args.learning_rate, "and reward factor:", args.reward_factor)

