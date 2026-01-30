import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
import random

# Import your existing modules
from train import Diffusion, save_images
from unet import UNetModel
from feature_extractor import ImageEncoder
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer
import copy
from torch.nn import DataParallel

# Import the CSV dataset
from csv_dataset import CSVHandwritingDataset


def save_single_images(images, path, args):
    """Save individual images instead of a grid"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for idx, img in enumerate(images):
        img_pil = torchvision.transforms.ToPILImage()(img)
        if args.color == False:
            img_pil = img_pil.convert('L')
        else:
            img_pil = img_pil.convert('RGB')
        
        # Extract filename without extension
        base_path = os.path.splitext(path)[0]
        img_pil.save(f'{base_path}_{idx}.png')


def load_style_images_from_csv(csv_data, style_id, transform, device, num_images=5):
    """Load style reference images for a given writer ID from CSV data"""
    # Filter images by writer ID
    style_samples = [item for item in csv_data if item[2] == style_id]
    
    if len(style_samples) < num_images:
        # If not enough samples, repeat some
        style_samples = style_samples * (num_images // len(style_samples) + 1)
    
    # Select random samples
    selected_samples = random.sample(style_samples, num_images)
    
    style_images = []
    for img, _, _, _ in selected_samples:
        # Apply transform if image is PIL, otherwise it's already a tensor
        if transform is not None and isinstance(img, Image.Image):
            img_tensor = transform(img)
        else:
            img_tensor = img
        style_images.append(img_tensor)
    
    # Stack into tensor [5, 3, 64, 256]
    style_tensor = torch.stack(style_images).to(device)
    return style_tensor


def test_single_word(diffusion, model, vae, tokenizer, feature_extractor, 
                     noise_scheduler, transform, args, word, style_id, train_data):
    """Generate a single word in a specific style"""
    
    print(f'\n=== Generating: "{word}" in style {style_id} ===')
    
    labels = torch.tensor([style_id]).long().to(args.device)
    
    # Load style reference images from the dataset
    style_images = load_style_images_from_csv(
        train_data.data, style_id, transform, args.device, num_images=5
    )
    
    # Extract features: [5, 3, 64, 256] -> [5, 1280]
    style_features = feature_extractor(style_images)
    
    # Prepare text
    text_features = tokenizer([word], padding="max_length", truncation=True, 
                             return_tensors="pt", max_length=40).to(args.device)
    
    # Generate without the complex file-loading logic
    model.eval()
    with torch.no_grad():
        if args.latent:
            x = torch.randn((1, 4, args.img_size[0] // 8, args.img_size[1] // 8)).to(args.device)
        else:
            x = torch.randn((1, 3, args.img_size[0], args.img_size[1])).to(args.device)
        
        # Diffusion sampling loop
        noise_scheduler.set_timesteps(50)
        for time in noise_scheduler.timesteps:
            t = torch.ones(1).long().to(args.device) * time.item()
            
            with torch.no_grad():
                # Reshape style features properly for the model
                # Expected shape: [batch_size, 5*feature_dim] = [1, 5*1280] if using all 5 images
                # But the model uses mean, so: [batch_size, feature_dim] = [1, 1280]
                # Looking at unet.py line 1318: y = y.reshape(b, 5, -1) then takes mean
                # So we need to pass [1, 5, 1280] which becomes [5, 1280] when reshaped
                
                # The style_features from feature_extractor is [5, 1280]
                # We need to reshape it to [1, 5*1280] = [1, 6400] for the model
                style_feat_reshaped = style_features.reshape(1, -1)  # [1, 6400]
                
                noisy_residual = model(
                    x, t, text_features, labels,
                    original_images=style_images.unsqueeze(0),
                    mix_rate=args.mix_rate,
                    style_extractor=style_feat_reshaped
                )
                prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                x = prev_noisy_sample
        
        # Decode from latent space
        if args.latent:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu()
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            image = x.cpu()
    
    # Save the result
    output_path = os.path.join(args.save_path, 'samples', f'{word}_style_{style_id}.png')
    save_single_images(image, output_path, args)
    print(f'Saved to: {output_path}')


def test_multiple_styles(diffusion, model, vae, tokenizer, feature_extractor,
                        noise_scheduler, transform, args, word, train_data, num_styles=16):
    """Generate the same word in multiple styles"""
    
    print(f'\n=== Generating "{word}" in {num_styles} different styles ===')
    
    # Sample random style IDs
    max_style = args.num_style_classes
    style_ids = random.sample(range(max_style), min(num_styles, max_style))
    
    all_images = []
    
    for style_id in style_ids:
        labels = torch.tensor([style_id]).long().to(args.device)
        
        # Load style reference images
        style_images = load_style_images_from_csv(
            train_data.data, style_id, transform, args.device, num_images=5
        )
        
        style_features = feature_extractor(style_images)
        
        # Prepare text
        text_features = tokenizer([word], padding="max_length", truncation=True,
                                 return_tensors="pt", max_length=40).to(args.device)
        
        # Generate
        model.eval()
        with torch.no_grad():
            if args.latent:
                x = torch.randn((1, 4, args.img_size[0] // 8, args.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((1, 3, args.img_size[0], args.img_size[1])).to(args.device)
            
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                t = torch.ones(1).long().to(args.device) * time.item()
                
                with torch.no_grad():
                    # Reshape: [5, 1280] -> [1, 6400]
                    style_feat_reshaped = style_features.reshape(1, -1)
                    
                    noisy_residual = model(
                        x, t, text_features, labels,
                        original_images=style_images.unsqueeze(0),
                        mix_rate=args.mix_rate,
                        style_extractor=style_feat_reshaped
                    )
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample
            
            # Decode
            if args.latent:
                latents = 1 / 0.18215 * x
                image = vae.module.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu()
            else:
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                image = x.cpu()
            
            all_images.append(image)
    
    # Stack all images
    all_images = torch.cat(all_images, dim=0)
    
    # Save as grid
    output_path = os.path.join(args.save_path, 'samples', f'{word}_multi_style.jpg')
    save_images(all_images, output_path, args)
    print(f'Saved grid to: {output_path}')


def test_from_test_set(diffusion, model, vae, test_loader, tokenizer, 
                       feature_extractor, noise_scheduler, args):
    """Generate samples from the test dataset"""
    
    print(f'\n=== Generating samples from test set ===')
    
    # Get one batch from test set
    test_batch = next(iter(test_loader))
    images, transcr, wid, style_images, img_path, cor_im = test_batch
    
    # Take first 16 samples
    n_samples = min(16, len(images))
    transcr = transcr[:n_samples]
    wid = wid[:n_samples]
    style_images = style_images[:n_samples]
    
    print(f'Generating {n_samples} samples...')
    print(f'Sample texts: {transcr[:5]}')
    
    # Generate images
    sampled_images = diffusion.sampling_loader(
        model=model,
        test_loader=test_loader,
        vae=vae,
        n=n_samples,
        x_text=None,
        labels=None,
        args=args,
        style_extractor=feature_extractor,
        noise_scheduler=noise_scheduler,
        transform=None,
        character_classes=None,
        tokenizer=tokenizer,
        text_encoder=None
    )
    
    # Save grid
    output_path = os.path.join(args.save_path, 'samples', 'test_set_samples.jpg')
    save_images(sampled_images, output_path, args)
    print(f'Saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Test DiffusionPen Model')
    
    # Dataset parameters
    parser.add_argument('--csv_path', type=str, default='train_hand.csv')
    parser.add_argument('--data_folder', type=str, default='./')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Paths
    parser.add_argument('--save_path', type=str, default='./diffusionpen_csv_model',
                       help='Path where model was saved during training')
    parser.add_argument('--style_path', type=str, default='./style_models/style_encoder.pth')
    parser.add_argument('--stable_dif_path', type=str, default='CompVis/stable-diffusion-v1-4',
                       help='Use CompVis/stable-diffusion-v1-4 (no auth required) or stabilityai/stable-diffusion-2-1-base (requires auth)')
    
    # Test options
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test_mode', type=str, default='single_word',
                       choices=['single_word', 'multiple_styles', 'test_set'],
                       help='What kind of test to run')
    parser.add_argument('--test_word', type=str, default='hello',
                       help='Word to generate (for single_word and multiple_styles modes)')
    parser.add_argument('--style_id', type=int, default=0,
                       help='Style ID to use (for single_word mode)')
    
    # Model options
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    
    args = parser.parse_args()
    args.img_size = (64, 256)
    args.dataset = 'csv_handwriting'
    
    print(f'\n=== DiffusionPen Testing ===')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Using device: {args.device}')
    print(f'Model path: {args.save_path}')
    print(f'Test mode: {args.test_mode}')
    
    # Create output directory
    os.makedirs(os.path.join(args.save_path, 'samples'), exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset to get style classes
    print(f'\n=== Loading Dataset Info ===')
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
    args.num_style_classes = style_classes
    print(f'Number of style classes: {style_classes}')
    
    # Setup device
    if args.dataparallel:
        device_ids = [0, 1]
        print(f'Using DataParallel with devices: {device_ids}')
    else:
        idx = int(''.join(filter(str.isdigit, args.device))) if any(c.isdigit() for c in args.device) else 0
        device_ids = [idx]
    
    # Text encoder
    print('\n=== Loading Models ===')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)
    
    # Character classes
    character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    vocab_size = len(character_classes)
    
    # UNet model
    print('Loading UNet...')
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
    
    # Load trained weights - try EMA first, fall back to regular checkpoint
    ema_path = os.path.join(args.save_path, 'models', 'ema_ckpt.pt')
    ckpt_path = os.path.join(args.save_path, 'models', 'ckpt.pt')
    
    if os.path.exists(ema_path):
        print(f'Loading EMA checkpoint: {ema_path}')
        unet.load_state_dict(torch.load(ema_path, map_location=args.device))
    elif os.path.exists(ckpt_path):
        print(f'Loading regular checkpoint: {ckpt_path}')
        unet.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    else:
        raise FileNotFoundError(f'No checkpoint found at {args.save_path}/models/')
    
    unet.eval()
    print('UNet loaded successfully')
    
    # VAE
    if args.latent:
        print('Loading VAE...')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
        vae.eval()
    else:
        vae = None
    
    # DDIM scheduler
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
    
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    # Diffusion
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    print('\nAll models loaded successfully!')
    
    # Run test based on mode
    if args.test_mode == 'single_word':
        test_single_word(
            diffusion, unet, vae, tokenizer, feature_extractor,
            ddim, transform, args, args.test_word, args.style_id, train_data
        )
    
    elif args.test_mode == 'multiple_styles':
        test_multiple_styles(
            diffusion, unet, vae, tokenizer, feature_extractor,
            ddim, transform, args, args.test_word, train_data, num_styles=16
        )
    
    elif args.test_mode == 'test_set':
        # Load test data
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
        
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        test_from_test_set(
            diffusion, unet, vae, test_loader, tokenizer,
            feature_extractor, ddim, args
        )
    
    print('\n=== Testing Complete ===')


if __name__ == "__main__":
    main()