import torch
from train import Diffusion
from unet import UNetModel
from diffusers import AutoencoderKL, DDIMScheduler
from feature_extractor import ImageEncoder
from transformers import CanineModel, CanineTokenizer
import argparse
from torchvision import transforms
import copy

args = argparse.Namespace(
    device='cuda:0',
    latent=True,
    img_feat=False,
    img_size=(64, 256),
    channels=4,
    emb_dim=320,
    num_heads=4,
    num_res_blocks=1,
    interpolation=False,
    mix_rate=None,
    model_name='diffusionpen',
    color=True
)

# Load your trained model
style_classes = 410  # adjust to your number of writers

tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
text_encoder = CanineModel.from_pretrained("google/canine-c").to(args.device)

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
    vocab_size=79,
    text_encoder=text_encoder,
    args=args
).to(args.device)

# # Load checkpoint
# unet.load_state_dict(torch.load('./my_handwriting_model_v2/models/ema_ckpt.pt', map_location=args.device))
# unet.eval()

# Load checkpoint (handle nested DataParallel wrapper)
checkpoint = torch.load('./my_handwriting_model_v2/models/ema_ckpt.pt', map_location=args.device)

# Remove 'module.' prefix from ALL nested levels
from collections import OrderedDict
new_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    # Remove all instances of 'module.'
    new_key = k.replace('module.', '')
    new_checkpoint[new_key] = v

# Load with strict=False to ignore size mismatches in label_emb
unet.load_state_dict(new_checkpoint, strict=False)
unet.eval()

# Load VAE
vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae").to(args.device)
vae.eval()

# # Load style encoder
feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0).to(args.device)
# feature_extractor.load_state_dict(torch.load('./style_models/mixed_csv_mobilenetv2_100.pth', map_location=args.device))
# feature_extractor.eval()

# Load style encoder (handle DataParallel)
style_checkpoint = torch.load('./style_models/mixed_csv_mobilenetv2_100.pth', map_location=args.device)
new_style_checkpoint = OrderedDict()
for k, v in style_checkpoint.items():
    if k.startswith('module.'):
        new_style_checkpoint[k[7:]] = v
    else:
        new_style_checkpoint[k] = v

feature_extractor.load_state_dict(new_style_checkpoint, strict=False)
feature_extractor.eval()

# DDIM with MORE steps
ddim = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="scheduler")
ddim.set_timesteps(200)  # Increase from 50 to 200

# Generate
diffusion = Diffusion(img_size=args.img_size, args=args)

# Load some style images from your dataset
import pandas as pd
from PIL import Image
import json

df = pd.read_csv('../cleaned-fidel/train_hand.csv')

# Load the writer mapping from training
with open('./writers_dict_train_csv.json', 'r') as f:
    writer_to_id = json.load(f)

# Convert keys to appropriate type (string or int)
# Check if keys are strings or ints in the file
first_key = list(writer_to_id.keys())[0]
if isinstance(first_key, str):
    writer_to_id = {str(k): int(v) for k, v in writer_to_id.items()}
else:
    writer_to_id = {int(k): int(v) for k, v in writer_to_id.items()}

print(f"Available writers: {list(writer_to_id.keys())[:10]}...")  # Show first 10

# Find a valid writer with at least 5 images
writer_counts = df['writer'].value_counts()
valid_writers = writer_counts[writer_counts >= 5].index.tolist()

# Find first valid writer that's in our training dict
valid_writer = None
for w in valid_writers:
    try:
        # Try both string and int versions
        if str(w) in writer_to_id:
            valid_writer = w
            break
        elif w != 'unknown' and int(w) in writer_to_id:
            valid_writer = w
            break
    except (ValueError, TypeError):
        # Skip non-numeric writers like 'unknown'
        continue

if valid_writer is None:
    print("ERROR: No valid writer found!")
    exit()

print(f"Using writer ID: {valid_writer}")

# Get the mapped numeric ID
try:
    if str(valid_writer) in writer_to_id:
        writer_numeric_id = writer_to_id[str(valid_writer)]
    else:
        writer_numeric_id = writer_to_id[int(valid_writer)]
except (ValueError, KeyError) as e:
    print(f"ERROR: Could not map writer ID {valid_writer}: {e}")
    exit()

print(f"Mapped to numeric ID: {writer_numeric_id}")

# Get 5 images from this writer
writer_images = df[df['writer'] == valid_writer].head(5)

style_imgs = []
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

for _, row in writer_images.iterrows():
    img_path = f"../cleaned-fidel/Fidel_data/{row['image_filename']}"
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 64))
        img_tensor = transform(img)
        style_imgs.append(img_tensor)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue

if len(style_imgs) == 0:
    print("ERROR: No style images loaded!")
    exit()

print(f"Loaded {len(style_imgs)} style images")

style_batch = torch.stack(style_imgs).to(args.device)

# Extract style features
with torch.no_grad():
    style_features = feature_extractor(style_batch)
    style_features = style_features.reshape(1, -1)

# Generate
diffusion = Diffusion(img_size=args.img_size, args=args)

# Test with a word
text = "ሰላም"
labels = torch.tensor([writer_numeric_id]).to(args.device)  # Use numeric ID

# Enable style features
args.img_feat = True

with torch.no_grad():
    samples = diffusion.sampling(
        model=unet,
        vae=vae,
        n=1,
        x_text=text,
        labels=labels,
        args=args,
        style_extractor=style_features,
        noise_scheduler=ddim,
        transform=transform,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )

from torchvision.utils import save_image
save_image(samples[0], 'test_output_200steps.png')
print("Saved test_output_200steps.png")