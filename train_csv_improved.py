import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

# Import your existing modules
from train import (
    train, Diffusion, EMA, setup_logging
)
from unet import UNetModel
from feature_extractor import ImageEncoder
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer
import copy
from torch.nn import DataParallel
from csv_dataset import CSVHandwritingDataset
from tqdm import tqdm
import numpy as np


def train_improved(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    """Improved training function with better monitoring"""
    model.train()
    print('Training started....')
    
    # Track best loss for early stopping
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch+1}/{args.epochs} ===')
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
        
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            # Tokenize text
            text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            # Extract style features
            if style_extractor is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                style_features = style_extractor(reshaped_images)
            else:
                style_features = None

            # Encode to latent space
            if args.latent == True:
                images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
            
            # Add noise
            noise = torch.randn(images.shape).to(images.device)
            num_train_timesteps = diffusion.noise_steps
            
            timesteps = torch.randint(
                0, num_train_timesteps,
                (images.shape[0],), device=images.device
            ).long()
            
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Predict noise
            predicted_noise = model(noisy_images, timesteps=timesteps, context=text_features, y=s_id, style_extractor=style_features)
            
            # Calculate loss
            loss = mse_loss(noise, predicted_noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            ema.step_ema(ema_model, model)

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Update learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}')
        
        # Save samples every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print('Generating samples...')
            labels = torch.arange(min(16, num_classes)).long().to(args.device)
            n = len(labels)
            
            # Sample from test set
            test_batch = next(iter(test_loader))
            sample_transcr = test_batch[1][:n]
            
            with torch.no_grad():
                model.eval()
                
                # Get style images
                style_imgs = test_batch[3][:n].to(args.device)
                reshaped_style = style_imgs.reshape(-1, 3, 64, 256)
                style_feats = style_extractor(reshaped_style) if style_extractor else None
                
                # Tokenize
                text_feats = tokenizer(sample_transcr, padding="max_length", truncation=True, 
                                      return_tensors="pt", max_length=40).to(args.device)
                
                # Generate
                if args.latent:
                    x = torch.randn((n, 4, 8, 32)).to(args.device)
                else:
                    x = torch.randn((n, 3, 64, 256)).to(args.device)
                
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:
                    t = torch.ones(n).long().to(args.device) * time.item()
                    noisy_residual = model(x, timesteps=t, context=text_feats, y=labels, style_extractor=style_feats)
                    x = noise_scheduler.step(noisy_residual, time, x).prev_sample
                
                # Decode
                if args.latent:
                    latents = 1 / 0.18215 * x
                    image = vae.module.decode(latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                else:
                    image = (x.clamp(-1, 1) + 1) / 2
                
                # Save
                save_path = os.path.join(args.save_path, 'images', f'epoch_{epoch:04d}.jpg')
                torchvision.utils.save_image(image, save_path, nrow=4)
                print(f'Saved samples to {save_path}')
                
                model.train()
            
            # Save checkpoints
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim.pt"))
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "best_ema_ckpt.pt"))
                print(f'✓ New best model saved! Loss: {best_loss:.4f}')
            else:
                patience_counter += 1
                print(f'No improvement for {patience_counter} epochs (best: {best_loss:.4f})')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {patience} epochs without improvement')
                print(f'Best loss: {best_loss:.4f}')
                break


def main():
    parser = argparse.ArgumentParser(description='Train DiffusionPen with Improved Settings')
    
    # Dataset parameters
    parser.add_argument('--csv_path', type=str, default='train_hand.csv')
    parser.add_argument('--data_folder', type=str, default='./')
    
    # Training parameters - IMPROVED DEFAULTS
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.00005)  # Lower LR
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--level', type=str, default='word')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    
    # Paths
    parser.add_argument('--save_path', type=str, default='./diffusionpen_improved')
    parser.add_argument('--style_path', type=str, default='./style_models/style_encoder.pth')
    parser.add_argument('--stable_dif_path', type=str, default='CompVis/stable-diffusion-v1-4')
    
    # Options
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--use_lr_scheduler', type=bool, default=True)
    
    args = parser.parse_args()
    args.img_size = (64, 256)
    args.dataset = 'csv_handwriting'
    
    print(f'\n=== DiffusionPen Improved Training ===')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Device: {args.device}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Using LR scheduler: {args.use_lr_scheduler}')
    
    # Create save directories
    setup_logging(args)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    print(f'\n=== Loading Dataset ===')
    train_data = CSVHandwritingDataset(
        basefolder=args.data_folder,
        subset='train',
        segmentation_level='word',
        fixed_size=(64, 256),
        tokenizer=None,
        text_encoder=None,
        feat_extractor=None,
        transforms=transform,
        args=args,
        csv_path=args.csv_path
    )
    
    style_classes = train_data.wclasses
    print(f'Total samples: {len(train_data)}')
    print(f'Unique writers: {style_classes}')
    
    # Test dataset
    test_data = CSVHandwritingDataset(
        basefolder=args.data_folder,
        subset='test',
        segmentation_level='word',
        fixed_size=(64, 256),
        tokenizer=None,
        text_encoder=None,
        feat_extractor=None,
        transforms=transform,
        args=args,
        csv_path=args.csv_path
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup device
    if args.dataparallel:
        device_ids = [0, 1]
    else:
        idx = int(''.join(filter(str.isdigit, args.device))) if any(c.isdigit() for c in args.device) else 0
        device_ids = [idx]
    
    # Text encoder
    print(f'\n=== Loading Models ===')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)
    
    # Character classes
    character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    vocab_size = len(character_classes)
    
    # UNet
    print('Creating UNet...')
    unet = UNetModel(
        image_size=args.img_size,
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=style_classes,
        context_dim=args.emb_dim,
        vocab_size=vocab_size,
        text_encoder=text_encoder,
        args=args
    )
    
    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)
    
    # Optimizer with lower learning rate
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler (cosine annealing)
    if args.use_lr_scheduler:
        total_steps = len(train_loader) * args.epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
        print(f'Using Cosine Annealing LR scheduler (total steps: {total_steps})')
    else:
        lr_scheduler = None
    
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    # EMA
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    # Load checkpoint if requested
    if args.load_check:
        print('Loading checkpoint...')
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
    
    # VAE
    if args.latent:
        print('Loading VAE...')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None
    
    # Scheduler
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    # Style encoder
    print('Loading style encoder...')
    feature_extractor = ImageEncoder(
        model_name='mobilenetv2_100',
        num_classes=0,
        pretrained=True,
        trainable=True
    )
    
    if os.path.exists(args.style_path):
        state_dict = torch.load(args.style_path, map_location=args.device)
        model_dict = feature_extractor.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        feature_extractor.load_state_dict(model_dict)
        print(f'Loaded style encoder from {args.style_path}')
    else:
        print('Warning: Style encoder not found, using random initialization')
    
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    # Training
    print(f'\n=== Starting Training ===')
    train_improved(
        diffusion=diffusion,
        model=unet,
        ema=ema,
        ema_model=ema_model,
        vae=vae,
        optimizer=optimizer,
        mse_loss=mse_loss,
        loader=train_loader,
        test_loader=test_loader,
        num_classes=style_classes,
        style_extractor=feature_extractor,
        vocab_size=vocab_size,
        noise_scheduler=ddim,
        transforms=transform,
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        lr_scheduler=lr_scheduler
    )
    
    print('\n=== Training Complete ===')


if __name__ == "__main__":
    main()