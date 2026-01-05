"""
Strawberry Dataset for PyTorch
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class StrawberryDataset(Dataset):
    """
    PyTorch Dataset for Strawberry Ripeness Classification.
    
    Labels:
        0: Unripe (green)
        1: Semi-ripe (transitioning)
        2: Ripe (ready to eat)
        3: Overripe (past peak)
    """
    
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        """
        Args:
            csv_file: Path to CSV with columns ['image', 'label']
            image_dir: Directory containing strawberry images
            transform: Optional torchvision transforms
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations.columns = ['image', 'label']
        
        # Handle labels stored as strings like "[0]"
        self.annotations['label'] = self.annotations['label'].apply(
            lambda x: eval(x)[0] if isinstance(x, str) and x.startswith('[') else int(x)
        )
        
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train: bool = True):
    """
    Get data transforms for training or validation.
    
    Args:
        train: If True, include data augmentation
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def create_data_splits(csv_file: str, output_dir: str = '.', test_size: float = 0.3):
    """
    Split dataset into train/val/test sets (70/15/15).
    
    Args:
        csv_file: Path to full labels CSV
        output_dir: Directory to save split CSVs
        test_size: Fraction for test+val combined
    """
    df = pd.read_csv(csv_file)
    df.columns = ['image', 'label']
    df['label'] = df['label'].apply(
        lambda x: int(eval(x)[0]) if isinstance(x, str) else int(x[0]) if isinstance(x, list) else int(x)
    )

    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
    )

    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_labels.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_dataloaders(image_dir: str, batch_size: int = 32, num_workers: int = 4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        image_dir: Directory containing images
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = StrawberryDataset(
        csv_file='train_labels.csv',
        image_dir=image_dir,
        transform=get_transforms(train=True)
    )
    val_dataset = StrawberryDataset(
        csv_file='val_labels.csv',
        image_dir=image_dir,
        transform=get_transforms(train=False)
    )
    test_dataset = StrawberryDataset(
        csv_file='test_labels.csv',
        image_dir=image_dir,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# Class mapping for visualization
CLASS_NAMES = {
    0: "Unripe",
    1: "Semi-ripe", 
    2: "Ripe",
    3: "Overripe"
}


if __name__ == "__main__":
    # Example usage
    create_data_splits('labels_fraises.csv')
    train_loader, val_loader, test_loader = get_dataloaders('images/')
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
