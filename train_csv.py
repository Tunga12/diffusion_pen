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

# Import the new CSV dataset
from csv_dataset import CSVHandwritingDataset


def main():
    parser = argparse.ArgumentParser(description='Train DiffusionPen on CSV Dataset')
    
    # Dataset parameters
    parser.add_argument('--csv_path', type=str, default='train_hand.csv', help='Path to CSV file')
    parser.add_argument('--data_folder', type=str, default='./', help='Root folder containing images')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64, help='Reduce if running out of memory')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--level', type=str, default='word', help='word or line')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    
    # Paths
    parser.add_argument('--save_path', type=str, default='./diffusionpen_csv_model')
    parser.add_argument('--style_path', type=str, default='./style_models/style_encoder.pth', 
                        help='Path to pretrained style encoder (optional)')
    parser.add_argument('--stable_dif_path', type=str, default='stabilityai/stable-diffusion-2-1-base',
                        help='Can use: stabilityai/stable-diffusion-2-1-base or CompVis/stable-diffusion-v1-4')
    
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
    parser.add_argument('--train_mode', type=str, default='train', help='train or sampling')
    
    args = parser.parse_args()
    args.img_size = (64, 256)
    args.dataset = 'csv_handwriting'
    
    print(f'PyTorch version: {torch.__version__}')
    print(f'Using device: {args.device}')
    
    # Create save directories
    setup_logging(args)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CSV dataset
    print(f'\n=== Loading CSV Dataset ===')
    print(f'CSV path: {args.csv_path}')
    print(f'Data folder: {args.data_folder}')
    
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
    
    # Get number of style classes (unique writers)
    style_classes = train_data.wclasses
    print(f'\nDataset statistics:')
    print(f'  - Total training samples: {len(train_data)}')
    print(f'  - Number of unique writers: {style_classes}')
    
    # Create test dataset (same CSV, different subset)
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
    print(f'  - Test samples: {len(test_data)}')
    
    # Create data loaders
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
    
    print(f'\n=== Model Setup ===')
    
    # Character classes
    character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    vocab_size = len(character_classes)
    print(f'Vocabulary size: {vocab_size}')
    
    # Setup device
    if args.dataparallel:
        device_ids = [0, 1]  # Adjust based on your GPUs
        print(f'Using DataParallel with devices: {device_ids}')
    else:
        idx = int(''.join(filter(str.isdigit, args.device))) if any(c.isdigit() for c in args.device) else 0
        device_ids = [idx]
    
    # Text encoder (CANINE)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)
    
    # UNet model
    print('Creating UNet model...')
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
    
    print(f'UNet parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}')
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=0.0001)
    mse_loss = nn.MSELoss()
    
    # Diffusion
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    # EMA
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    # VAE
    if args.latent:
        print('Loading VAE from Stable Diffusion...')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None
    
    # DDIM scheduler
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    # Style encoder (feature extractor)
    print('Setting up style encoder...')
    feature_extractor = ImageEncoder(
        model_name='mobilenetv2_100',
        num_classes=0,
        pretrained=True,
        trainable=True
    )
    
    # Load pretrained style encoder if available
    if os.path.exists(args.style_path):
        print(f'Loading pretrained style encoder from {args.style_path}')
        state_dict = torch.load(args.style_path, map_location=args.device)
        model_dict = feature_extractor.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        feature_extractor.load_state_dict(model_dict)
    else:
        print(f'Warning: Style encoder not found at {args.style_path}. Using randomly initialized weights.')
        print('Consider training a style encoder first using style_encoder_train.py')
    
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    # Load checkpoint if requested
    if args.load_check:
        print('Loading checkpoint...')
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Checkpoint loaded successfully')
    
    # Training
    print(f'\n=== Starting Training ===')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Save path: {args.save_path}')
    
    train(
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
        lr_scheduler=None
    )
    
    print('\n=== Training Complete ===')


if __name__ == "__main__":
    main()