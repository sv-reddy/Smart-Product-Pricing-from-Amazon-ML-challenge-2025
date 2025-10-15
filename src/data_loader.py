import os
import pandas as pd
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm
import time

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Field.*has no effect.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCache:
    """Efficient image caching system to avoid repeated downloads"""
    
    def __init__(self, cache_dir: str = "image_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.jpg"
    
    def get_image(self, url: str, max_retries: int = 3) -> Optional[Image.Image]:
        """Download and cache image with retry logic"""
        cache_path = self._get_cache_path(url)
        
        # Try to load from cache first
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load cached image {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)
        
        # Download image with retries
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Save to cache
                img.save(cache_path, 'JPEG', quality=85, optimize=True)
                return img
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download image after {max_retries} attempts: {url} - {e}")
                    return None
                time.sleep(1)  # Wait before retry
        
        return None

class TextProcessor:
    """Advanced text preprocessing for catalog content"""
    
    def __init__(self, tokenizer_name: str = "distilbert-base-uncased", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Extract structured information
        text = self._extract_structured_info(text)
        
        # Basic cleaning
        text = text.strip()
        return text[:2000]  # Limit length to prevent memory issues
    
    def _extract_structured_info(self, text: str) -> str:
        """Extract and format structured information from catalog content"""
        # Extract item name
        item_name_match = re.search(r'Item Name:\s*([^\n]+)', text)
        item_name = item_name_match.group(1) if item_name_match else ""
        
        # Extract bullet points
        bullet_points = re.findall(r'Bullet Point \d+:\s*([^\n]+)', text)
        
        # Extract value and unit
        value_match = re.search(r'Value:\s*([^\n]+)', text)
        unit_match = re.search(r'Unit:\s*([^\n]+)', text)
        
        value = value_match.group(1) if value_match else ""
        unit = unit_match.group(1) if unit_match else ""
        
        # Extract product description
        desc_match = re.search(r'Product Description:\s*([^\n]+)', text)
        description = desc_match.group(1) if desc_match else ""
        
        # Combine structured information
        structured_text = f"{item_name}. "
        
        if value and unit:
            structured_text += f"Size: {value} {unit}. "
        
        if bullet_points:
            structured_text += "Features: " + " ".join(bullet_points[:3]) + ". "  # Limit to 3 bullet points
        
        if description:
            structured_text += f"Description: {description}"
        
        return structured_text.strip()
    
    def tokenize(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Tokenize text using the transformer tokenizer"""
        clean_text = self.clean_text(text)
        
        # Use instance max_length if not provided
        if max_length is None:
            max_length = self.max_length
        
        encoding = self.tokenizer(
            clean_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

class ImageProcessor:
    """Image preprocessing with augmentations"""
    
    def __init__(self, image_size: int = 224, is_training: bool = True):
        self.image_size = image_size
        self.is_training = is_training
        
        # Training augmentations
        if is_training:
            self.transform = A.Compose([
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation/test augmentations
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process PIL image and return tensor"""
        if image is None:
            # Return black image if download failed
            return torch.zeros(3, self.image_size, self.image_size)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply transformations
        transformed = self.transform(image=image_array)
        return transformed['image']

class ProductDataset(Dataset):
    """Dataset class for product data with text and images"""
    
    def __init__(
        self,
        csv_path: str,
        text_processor: TextProcessor,
        image_processor: ImageProcessor,
        image_cache: ImageCache,
        max_samples: Optional[int] = None,
        is_training: bool = True
    ):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.head(max_samples)
            
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.image_cache = image_cache
        self.is_training = is_training
        
        # Log dataset info
        logger.info(f"Loaded dataset with {len(self.df)} samples from {csv_path}")
        if 'price' in self.df.columns:
            logger.info(f"Price range: {self.df['price'].min():.2f} - {self.df['price'].max():.2f}")
            
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Process text
        text_data = self.text_processor.tokenize(row['catalog_content'])
        
        # Process image
        image = self.image_cache.get_image(row['image_link'])
        image_tensor = self.image_processor.process_image(image)
        
        result = {
            'sample_id': torch.tensor(row['sample_id'], dtype=torch.long),
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'image': image_tensor
        }
        
        # Add price for training data
        if 'price' in row and pd.notna(row['price']):
            result['price'] = torch.tensor(float(row['price']), dtype=torch.float32)
        
        return result

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader"""
    result = {}
    
    # Stack tensors
    for key in ['sample_id', 'input_ids', 'attention_mask', 'image']:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    
    # Handle price (might not be present in test data)
    if 'price' in batch[0]:
        result['price'] = torch.stack([item['price'] for item in batch])
    
    return result

def create_data_loaders(
    train_csv: str,
    test_csv: str,
    batch_size: int = 16,
    num_workers: int = 2,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    image_size: int = 224,
    max_text_length: int = 256
) -> Tuple[DataLoader, DataLoader, TextProcessor, ImageCache]:
    """Create training and test data loaders"""
    
    # Initialize processors
    text_processor = TextProcessor()
    image_cache = ImageCache()
    
    # Create datasets
    train_image_processor = ImageProcessor(image_size=image_size, is_training=True)
    test_image_processor = ImageProcessor(image_size=image_size, is_training=False)
    
    train_dataset = ProductDataset(
        csv_path=train_csv,
        text_processor=text_processor,
        image_processor=train_image_processor,
        image_cache=image_cache,
        max_samples=max_train_samples,
        is_training=True
    )
    
    test_dataset = ProductDataset(
        csv_path=test_csv,
        text_processor=text_processor,
        image_processor=test_image_processor,
        image_cache=image_cache,
        max_samples=max_test_samples,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)} batches, Test: {len(test_loader)} batches")
    
    return train_loader, test_loader, text_processor, image_cache

if __name__ == "__main__":
    # Test data loading
    train_csv = "../dataset/train.csv"
    test_csv = "../dataset/test.csv"
    
    # Create small test loaders
    train_loader, test_loader, text_processor, image_cache = create_data_loaders(
        train_csv=train_csv,
        test_csv=test_csv,
        batch_size=4,
        max_train_samples=100,  # Test with small sample
        max_test_samples=50
    )
    
    # Test loading a batch
    print("Testing data loading...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        if batch_idx >= 2:  # Test only first few batches
            break
    
    print("Data loading test completed successfully!")