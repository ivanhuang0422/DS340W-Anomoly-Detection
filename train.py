import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import CombinedLoss, FocalLoss, SSIM  # Import all required losses
from data_loader import MVTecDRAEMTrainDataset
import os
import numpy as np

class AnomalyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.model = ReconstructiveSubNetwork().to(self.device)
        self.model_seg = DiscriminativeSubNetwork().to(self.device)
        
        # Adjusted learning rates and optimizer
        self.optimizer = optim.AdamW([  # Changed to AdamW
            {"params": self.model.parameters(), "lr": args.lr * 3},    # Increased reconstruction learning rate
            {"params": self.model_seg.parameters(), "lr": args.lr * 4} # Increased segmentation learning rate
        ], weight_decay=0.01)  # Added weight decay
        
        # Use OneCycleLR scheduler with higher max_lr
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[args.lr * 3, args.lr * 4],
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            pct_start=0.3  # Warm-up for first 30% of training
        )
        
        # Adjusted loss weights
        self.combined_loss = CombinedLoss(
            alpha=0.3,  # Reconstruction weight
            beta=0.3,   # SSIM weight
            gamma=0.4   # Segmentation weight (increased)
        ).to(self.device)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        self.model_seg.train()
        
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # Get data with augmentation
            images = batch["image"].to(self.device)
            aug_images = batch["augmented_image"].to(self.device)
            masks = batch["anomaly_mask"].to(self.device)
            
            # Apply additional augmentation
            if torch.rand(1) < 0.5:  # Random horizontal flip
                images = torch.flip(images, dims=[3])
                aug_images = torch.flip(aug_images, dims=[3])
                masks = torch.flip(masks, dims=[3])
            
            # Forward pass
            rec_images, features = self.model(aug_images)
            joined = torch.cat((rec_images.detach(), aug_images), dim=1)
            seg_out = self.model_seg(joined)
            seg_out_sm = torch.softmax(seg_out, dim=1)
            
            # Calculate main loss
            loss = self.combined_loss(rec_images, images, seg_out, masks)
            
            # Add feature matching loss
            with torch.no_grad():
                orig_features = self.model(images)[1]
            feature_loss = F.mse_loss(features, orig_features.detach())
            loss += 0.2 * feature_loss  # Increased feature loss weight
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Added gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model_seg.parameters(), 1.0)
            self.optimizer.step()
            
            # Update learning rates
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if i % 2 == 0:
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch, obj_name):
        checkpoint_path = os.path.join(
            self.args.checkpoint_path,
            f"DRAEM_test_{self.args.lr}_{self.args.epochs}_bs{self.args.bs}_{obj_name}_"
        )
        
        torch.save(self.model.state_dict(), checkpoint_path + ".pckl")
        torch.save(self.model_seg.state_dict(), checkpoint_path + "_seg.pckl")

def train_on_device(obj_names, args):
    for obj_name in obj_names:
        print(f"\nTraining on {obj_name}")
        print("="*50)
        
        # Initialize trainer
        trainer = AnomalyTrainer(args)
        
        # Setup data
        dataset = MVTecDRAEMTrainDataset(
            os.path.join(args.data_path, obj_name, "train/good/"),
            args.anomaly_source_path,
            resize_shape=[256, 256]
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=4
        )
        
        print(f"Dataset size: {len(dataset)} images")
        print(f"Batch size: {args.bs}")
        print(f"Steps per epoch: {len(dataloader)}")
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            loss = trainer.train_epoch(dataloader, epoch)
            print(f"Epoch {epoch+1} completed, Average Loss: {loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(epoch, obj_name)
                print(f"Checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--anomaly_source_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    # Calculate steps per epoch based on dataset size and batch size
    # This is a placeholder value - it will be updated when we know the actual dataset size
    args.steps_per_epoch = 1000  # Will be adjusted based on actual dataset size
    
    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
                ['transistor'], ['tile'], ['cable'], ['zipper'], ['toothbrush'],
                ['metal_nut'], ['hazelnut'], ['screw'], ['grid'], ['wood']]
    
    if args.obj_id == -1:
        picked_classes = [item[0] for item in obj_batch]
    else:
        picked_classes = obj_batch[args.obj_id]
    
    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)