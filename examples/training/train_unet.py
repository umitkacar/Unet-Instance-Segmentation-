"""
Complete U-Net Training Script
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from segmentation_models_pytorch.metrics import iou_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import os


class SegmentationDataset(Dataset):
    """Custom dataset for segmentation tasks"""

    def __init__(self, images, masks, transform=None):
        """
        Args:
            images: List of image paths or numpy arrays
            masks: List of mask paths or numpy arrays
            transform: Albumentations transform
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        # (In practice, load from file using PIL or cv2)
        image = self.images[idx]  # Shape: [H, W, 3]
        mask = self.masks[idx]     # Shape: [H, W, 1]

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_training_augmentation():
    """Define training augmentations"""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform


def get_validation_augmentation():
    """Define validation augmentations"""
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return val_transform


class Trainer:
    """Trainer class for U-Net model"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_iou = 0.0

        progress_bar = tqdm(self.train_loader, desc='Training')
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            iou = iou_score(outputs, masks, threshold=0.5)

            epoch_loss += loss.item()
            epoch_iou += iou.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou.item():.4f}'
            })

        return epoch_loss / len(self.train_loader), epoch_iou / len(self.train_loader)

    def validate(self):
        """Validate the model"""
        self.model.eval()

        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_f1 = 0.0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Metrics
                iou = iou_score(outputs, masks, threshold=0.5)
                f1 = f1_score(outputs, masks, threshold=0.5)

                epoch_loss += loss.item()
                epoch_iou += iou.item()
                epoch_f1 += f1.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou.item():.4f}',
                    'f1': f'{f1.item():.4f}'
                })

        avg_loss = epoch_loss / len(self.val_loader)
        avg_iou = epoch_iou / len(self.val_loader)
        avg_f1 = epoch_f1 / len(self.val_loader)

        return avg_loss, avg_iou, avg_f1

    def save_checkpoint(self, epoch, val_loss, val_iou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'last_checkpoint.pth'))

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_checkpoint.pth'))
            print(f"✓ Best model saved! (IoU: {val_iou:.4f})")

    def train(self, num_epochs):
        """Train the model for specified epochs"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("="*60)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Training
            train_loss, train_iou = self.train_epoch()

            # Validation
            val_loss, val_iou, val_f1 = self.validate()

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f} | Val F1: {val_f1:.4f}")

            # Save checkpoint
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, val_iou, is_best)

        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best Val IoU: {self.best_val_iou:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")


def main():
    """Main training function"""

    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )

    # Create dummy datasets (replace with actual data loading)
    print("Creating datasets...")
    # In practice: load actual images and masks from disk
    train_images = [np.random.rand(512, 512, 3).astype(np.float32) for _ in range(100)]
    train_masks = [np.random.randint(0, 2, (512, 512, 1)).astype(np.float32) for _ in range(100)]
    val_images = [np.random.rand(512, 512, 3).astype(np.float32) for _ in range(20)]
    val_masks = [np.random.randint(0, 2, (512, 512, 1)).astype(np.float32) for _ in range(20)]

    # Create datasets
    train_dataset = SegmentationDataset(
        train_images, train_masks,
        transform=get_training_augmentation()
    )
    val_dataset = SegmentationDataset(
        val_images, val_masks,
        transform=get_validation_augmentation()
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Loss function (combination of Dice + BCE)
    criterion = DiceLoss(mode='binary')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        save_dir='checkpoints'
    )

    # Train
    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
