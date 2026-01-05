"""
Training Script for Strawberry Ripeness Classification

Usage:
    python train.py --model cnn --epochs 50
    python train.py --model resnet18 --epochs 30 --lr 1e-4
"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders, CLASS_NAMES
from model import get_model, count_parameters


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def evaluate(model, test_loader, device):
    """Full evaluation with confusion matrix and classification report."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification report
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=list(CLASS_NAMES.values())))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix.png'")

    return all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to '{save_path}'")


def main(args):
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, num_classes=4, pretrained=True)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # TensorBoard
    writer = SummaryWriter(f'runs/{args.model}_{args.epochs}epochs')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'best_{args.model}.pt')
            print(f"💾 Saved best model (Val Acc: {val_acc:.2f}%)")
    
    writer.close()
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    
    # Load best model
    checkpoint = torch.load(f'best_{args.model}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    evaluate(model, test_loader, device)
    
    print("\n✅ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Strawberry Ripeness Classifier")
    
    # Model
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18'],
                        help='Model architecture (default: cnn)')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='images',
                        help='Directory containing images (default: images)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    main(args)
