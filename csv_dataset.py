import numpy as np 
import pandas as pd
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from PIL import Image, ImageOps
import json
import os
import string

class CSVHandwritingDataset(WordLineDataset):
    """
    Dataset loader for custom CSV handwriting data.
    Expected CSV format: image_filename,line_text,type,writer
    """
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args, csv_path='train_hand.csv'):
        self.csv_path = csv_path
        super().__init__(basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args)
        self.setname = 'CSV_Handwriting'
        self.word_path = basefolder
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feat_extractor = feat_extractor
        self.args = args
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        """
        Load data from CSV file and prepare it for training.
        """
        # Read CSV file
        df = pd.read_csv(self.csv_path)
        
        # Convert writer column to string to handle mixed types
        df['writer'] = df['writer'].astype(str).str.strip()
        # Remove any rows with NaN, empty, or non-numeric writers
        df = df[df['writer'].notna() & (df['writer'] != '') & (df['writer'] != 'nan')]
        # Keep only rows where writer is numeric
        df = df[df['writer'].str.isnumeric()]
        # Convert to int
        df['writer'] = df['writer'].astype(int)
        unique_writers = sorted(df['writer'].unique())
        
        # Create mapping from original writer IDs to consecutive indices (0, 1, 2, ...)
        # This ensures style classes are 0-indexed and consecutive
        writer_to_id = {int(writer): idx for idx, writer in enumerate(unique_writers)}
        
        # Save writer dictionary for reference
        dict_path = f'./writers_dict_{subset}_csv.json'
        with open(dict_path, 'w') as f:
            json.dump(writer_to_id, f)
        
        print(f"Loaded {len(df)} samples from {self.csv_path}")
        print(f"Number of unique writers: {len(unique_writers)}")
        print(f"Writer IDs range: {min(unique_writers)} to {max(unique_writers)}")
        
        # Split data based on subset (train/val/test)
        # You can customize this split logic based on your needs
        if subset == 'train':
            # Use 80% for training
            df_subset = df.iloc[:int(len(df) * 0.8)]
        elif subset == 'val':
            # Use 10% for validation
            df_subset = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
        elif subset == 'test':
            # Use 10% for testing
            df_subset = df.iloc[int(len(df) * 0.9):]
        else:
            df_subset = df
        
        print(f"Subset '{subset}' contains {len(df_subset)} samples")
        
        data = []
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        
        for i, row in df_subset.iterrows():
            if i % 1000 == 0:
                print(f'Processing: [{i}/{len(df_subset)} ({100. * i / len(df_subset):.0f}%)]')
            
            img_filename = row['image_filename']
            transcr = row['line_text']
            writer = row['writer']  # Already a number
            
            # Convert to consecutive 0-indexed ID
            writer_id = writer_to_id[writer]
            
            # Construct full image path
            img_path = os.path.join(self.word_path, img_filename)
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            try:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                
                # Handle punctuation differently
                if transcr in string.punctuation:
                    img = centered_PIL(img, (64, 256), border_value=255.0)
                else:
                    # Resize to height 64 keeping aspect ratio
                    img_width, img_height = img.size
                    img = img.resize((int(img_width * 64 / img_height), 64))
                    img_width, img_height = img.size
                    
                    if img_width < 256:
                        # Pad to 256 width
                        img = ImageOps.pad(img, size=(256, 64), color="white")
                    else:
                        # Reduce width to 256
                        while img_width > 256:
                            img = image_resize_PIL(img, width=img_width-20)
                            img_width, img_height = img.size
                        img = centered_PIL(img, (64, 256), border_value=255.0)
                
                # Clean transcription (remove extra spaces, etc.)
                transcr = transcr.strip()
                
                # Add to data list
                data.append((img, transcr, writer_id, img_path))
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        print(f'Successfully loaded {len(data)} samples for {subset}')
        
        return data