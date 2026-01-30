import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
from feature_extractor import ImageEncoder
import time
import pandas as pd
import json
import random
from torchvision import transforms


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class CSVStyleDataset(Dataset):
    """Dataset for loading handwriting images from CSV for style encoder training"""
    
    def __init__(self, csv_path, data_folder, subset='train', fixed_size=(64, 256), transforms=None):
        self.csv_path = csv_path
        self.data_folder = data_folder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Create writer mapping
        # unique_writers = sorted(df['writer'].unique())
        # Convert writer column to string to handle mixed types
        df['writer'] = df['writer'].astype(str).str.strip()
        # Remove any rows with NaN, empty, or non-numeric writers
        df = df[df['writer'].notna() & (df['writer'] != '') & (df['writer'] != 'nan')]
        # Keep only rows where writer is numeric
        df = df[df['writer'].str.isnumeric()]
        # Convert to int
        df['writer'] = df['writer'].astype(int)
        unique_writers = sorted(df['writer'].unique())

        self.writer_to_id = {int(writer): idx for idx, writer in enumerate(unique_writers)}

        # Save writer dictionary
        dict_path = f'./writers_dict_{subset}_csv.json'
        with open(dict_path, 'w') as f:
            json.dump(self.writer_to_id, f)
        
        print(f"Total samples in CSV: {len(df)}")
        print(f"Unique writers: {len(unique_writers)}")
        
        # Split data
        if subset == 'train':
            df_subset = df.iloc[:int(len(df) * 0.8)]
        elif subset == 'val':
            df_subset = df.iloc[int(len(df) * 0.8):]
        else:
            df_subset = df
        
        # Load all valid samples
        self.data_info = []
        for i, row in df_subset.iterrows():
            img_path = os.path.join(data_folder, row['image_filename'])
            if os.path.exists(img_path):
                writer_id = self.writer_to_id[row['writer']]
                transcr = row['line_text']
                self.data_info.append((img_path, writer_id, transcr))
        
        print(f"Loaded {len(self.data_info)} valid samples for {subset}")
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        img_path, wid, transcr = self.data_info[index]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize and pad
        img = self.process_image(img)
        
        # Get positive sample (same writer)
        positive_samples = [p for p in self.data_info if p[1] == wid and len(p[2]) > 3]
        if len(positive_samples) > 1:
            positive = random.choice([p for p in positive_samples if p[0] != img_path])
        else:
            positive = random.choice([p for p in self.data_info if p[1] == wid])
        
        # Get negative sample (different writer)
        negative_samples = [n for n in self.data_info if n[1] != wid and len(n[2]) > 3]
        negative = random.choice(negative_samples)
        
        # Load positive and negative images
        img_pos = Image.open(positive[0]).convert('RGB')
        img_neg = Image.open(negative[0]).convert('RGB')
        
        img_pos = self.process_image(img_pos)
        img_neg = self.process_image(img_neg)
        
        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)
        
        return img, transcr, wid, img_pos, img_neg, img_path
    
    def process_image(self, img):
        """Resize and pad image to fixed size"""
        fheight, fwidth = self.fixed_size
        
        # Resize to height while maintaining aspect ratio
        img_width, img_height = img.size
        img = img.resize((int(img_width * fheight / img_height), fheight))
        img_width, img_height = img.size
        
        if img_width < fwidth:
            # Pad to target width
            img = ImageOps.pad(img, size=(fwidth, fheight), color="white")
        else:
            # Reduce width if needed
            while img_width > fwidth:
                img_width = img_width - 20
                img = img.resize((img_width, fheight))
            # Center in target size
            result = Image.new('RGB', (fwidth, fheight), color=(255, 255, 255))
            offset = ((fwidth - img.width) // 2, 0)
            result.paste(img, offset)
            img = result
        
        return img
    
    def collate_fn(self, batch):
        img, transcr, wid, positive, negative, img_path = zip(*batch)
        
        images_batch = torch.stack(img)
        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)
        wid = torch.tensor(wid)
        
        return images_batch, transcr, wid, images_pos, images_neg, img_path


class Mixed_Encoder(nn.Module):
    """
    Encoder that outputs both classification logits and features for triplet loss
    """
    def __init__(self, model_name='mobilenetv2_100', num_classes=339, pretrained=True, trainable=True):
        super().__init__()
        import timm
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Get number of features
        if hasattr(self.model, 'num_features'):
            num_features = self.model.num_features
        else:
            num_features = 1280  # default for mobilenetv2
        
        self.classifier = nn.Linear(num_features, num_classes)
        
        for p in self.model.parameters():
            p.requires_grad = trainable
    
    def forward(self, x):
        features = self.model(x)
        pooled_features = self.global_pool(features).flatten(1)
        logits = self.classifier(pooled_features)
        return logits, pooled_features


def performance(pred, label):
    """Classification loss"""
    loss = nn.CrossEntropyLoss()
    loss = loss(pred, label)
    return loss


def train_epoch_mixed(train_loader, model, criterion_triplet, optimizer, device, args):
    """Training epoch with both classification and triplet loss"""
    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        img = data[0].to(device)
        wid = data[2].to(device)
        positive = data[3].to(device)
        negative = data[4].to(device)
        
        # Get logits and features
        anchor_logits, anchor_features = model(img)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
        
        # Calculate losses
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        loss = classification_loss + triplet_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        pbar.set_postfix(
            mixed_loss=loss_meter.avg,
            class_loss=loss_meter_class.avg,
            triplet_loss=loss_meter_triplet.avg
        )
        total += img.size(0)
    
    accuracy = n_corrects / total
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")
    print(f"Training Accuracy: {accuracy*100:.4f}%")
    return running_loss / total


def val_epoch_mixed(val_loader, model, criterion_triplet, device, args):
    """Validation epoch"""
    model.eval()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for i, data in enumerate(pbar):
            img = data[0].to(device)
            wid = data[2].to(device)
            positive = data[3].to(device)
            negative = data[4].to(device)
            
            anchor_logits, anchor_features = model(img)
            _, positive_features = model(positive)
            _, negative_features = model(negative)
            
            _, preds = torch.max(anchor_logits.data, 1)
            n_corrects += (preds == wid.data).sum().item()
            
            classification_loss = performance(anchor_logits, wid)
            triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
            
            loss = classification_loss + triplet_loss
            
            running_loss += loss.item()
            count = img.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(mixed_loss=loss_meter.avg)
            total += wid.size(0)
    
    accuracy = n_corrects / total
    print(f"Validation Loss: {running_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {accuracy*100:.4f}%")
    return running_loss / total


def train_mixed(model, train_loader, val_loader, criterion_triplet, optimizer, scheduler, device, args):
    """Main training loop"""
    best_loss = float('inf')
    
    for epoch_i in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch: {epoch_i+1}/{args.epochs}")
        print(f"{'='*50}")
        
        model.train()
        train_loss = train_epoch_mixed(train_loader, model, criterion_triplet, optimizer, device, args)
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_mixed(val_loader, model, criterion_triplet, device, args)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{args.save_path}/mixed_csv_{args.model}.pth')
            print(f"✓ Saved Best Model! (Loss: {best_loss:.4f})")
        
        scheduler.step(val_loss)


def main():
    parser = argparse.ArgumentParser(description='Train Style Encoder on CSV Dataset')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', help='Model architecture')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--data_folder', type=str, default='./', help='Root folder with images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--save_path', type=str, default='./style_models', help='Save path')
    parser.add_argument('--mode', type=str, default='mixed', help='Training mode (mixed)')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained backbone')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Style Encoder Training - CSV Dataset")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"CSV: {args.csv_path}")
    print(f"Data folder: {args.data_folder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    print("Loading training data...")
    train_data = CSVStyleDataset(
        csv_path=args.csv_path,
        data_folder=args.data_folder,
        subset='train',
        fixed_size=(64, 256),
        transforms=train_transform
    )
    
    print("Loading validation data...")
    val_data = CSVStyleDataset(
        csv_path=args.csv_path,
        data_folder=args.data_folder,
        subset='val',
        fixed_size=(64, 256),
        transforms=val_transform
    )
    
    style_classes = len(train_data.writer_to_id)
    print(f"\nNumber of style classes: {style_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_data.collate_fn
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_data.collate_fn
    )
    
    # Create model
    print(f"\nInitializing {args.model} model...")
    model = Mixed_Encoder(
        model_name=args.model,
        num_classes=style_classes,
        pretrained=args.pretrained,
        trainable=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )
    
    # Loss function
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # Train
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    train_mixed(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_triplet=criterion_triplet,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best model saved to: {args.save_path}/mixed_csv_{args.model}.pth")
    print("="*60)


if __name__ == '__main__':
    main()