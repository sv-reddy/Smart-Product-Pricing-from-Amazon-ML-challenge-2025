#!/usr/bin/env python3
"""
Continue Training Script for ML Challenge 2025
Loads existing model and continues training to achieve SMAPE < 40%

This script:
1. Loads the best existing model checkpoint
2. Applies advanced training techniques for SMAPE improvement
3. Uses refined hyperparameters for better convergence
4. Implements progressive learning rate strategies
"""

import os
import sys
import warnings
import torch
import logging
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Field.*has no effect.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add src to path
sys.path.append(str(Path(__file__).parent))

from multimodal_model import create_model, ModelConfig as ModelArchConfig
from train_model import Trainer, ModelConfig as TrainConfig, calculate_smape_metric, memory_cleanup
from data_loader import create_data_loaders, TextProcessor, ImageCache, ImageProcessor, ProductDataset, collate_fn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.optim as optim

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continue_training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuedTrainer:
    """
    Specialized trainer for continuing training from existing checkpoint
    Focuses on SMAPE improvement with advanced techniques
    """
    
    def __init__(self, checkpoint_path: str, target_smape: float = 40.0):
        self.checkpoint_path = checkpoint_path
        self.target_smape = target_smape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced training configuration
        self.train_config = TrainConfig()
        self.train_config.batch_size = 16          # Optimized for stability
        self.train_config.gradient_accumulation_steps = 4  # Effective batch = 64
        self.train_config.learning_rate = 1e-4     # Lower LR for fine-tuning
        self.train_config.weight_decay = 2e-4      # Increased regularization
        self.train_config.num_epochs = 10         # Extended training
        self.train_config.patience = 8             # More patience for gradual improvement
        self.train_config.min_delta = 0.01         # Smaller improvement threshold
        self.train_config.use_amp = True
        self.train_config.num_workers = 4

        # Model architecture configuration
        self.model_config = ModelArchConfig()  # This has the correct attributes for create_model
        
        self.best_smape = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_smape': [],
            'learning_rates': []
        }
        
        logger.info(f"🔄 Continued Trainer initialized")
        logger.info(f"🎯 Target SMAPE: < {target_smape}%")
        logger.info(f"💻 Device: {self.device}")
        
    def load_checkpoint(self):
        """Load existing model checkpoint and prepare for continued training"""
        logger.info(f"📂 Loading checkpoint: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Update config from checkpoint if available
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(self.train_config, key):
                    setattr(self.train_config, key, value)
        
        # Create model using the correct model config
        self.model = create_model(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Get previous best SMAPE if available
        self.previous_best = checkpoint.get('best_smape', 'Unknown')
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"📊 Previous best SMAPE: {self.previous_best}")
        logger.info(f"🎯 New target: < {self.target_smape}%")
        
        return checkpoint
    
    def setup_optimization(self):
        """Setup optimizers and schedulers for continued training"""
        # Optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            eps=1e-8
        )
        
        # Advanced scheduler for gradual improvement
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=True
        )
        
        # Secondary scheduler for warm restarts
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=2,
            eta_min=1e-7
        )
        
        # Loss function and scaler
        from train_model import SMAPELoss
        self.criterion = SMAPELoss()
        self.scaler = GradScaler()
        
        logger.info("⚙️ Optimization setup complete")
    
    def prepare_data(self, train_csv_path: str):
        """Prepare data loaders for continued training"""
        logger.info("📊 Preparing data for continued training...")
        
        # Use existing data splits if available
        train_split_path = Path("src/model_checkpoints/train_split.csv")
        val_split_path = Path("src/model_checkpoints/val_split.csv")
        
        if train_split_path.exists() and val_split_path.exists():
            logger.info("✅ Using existing train/val splits")
            train_csv = str(train_split_path)
            val_csv = str(val_split_path)
        else:
            logger.info("🔄 Creating new train/val splits")
            # Create new splits if needed
            df = pd.read_csv(train_csv_path)
            from sklearn.model_selection import train_test_split
            
            train_df, val_df = train_test_split(
                df, test_size=0.15, random_state=42,
                stratify=pd.cut(np.log1p(df['price']), bins=10, labels=False, duplicates='drop')
            )
            
            train_csv = "src/model_checkpoints/continue_train_split.csv"
            val_csv = "src/model_checkpoints/continue_val_split.csv"
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
        
        # Initialize processors
        text_processor = TextProcessor(self.model_config.text_model_name, self.model_config.max_text_length)
        train_image_proc = ImageProcessor(self.model_config.image_size, is_training=True)
        val_image_proc = ImageProcessor(self.model_config.image_size, is_training=False)
        image_cache = ImageCache()
        
        # Create datasets
        train_dataset = ProductDataset(train_csv, text_processor, train_image_proc, image_cache, is_training=True)
        val_dataset = ProductDataset(val_csv, text_processor, val_image_proc, image_cache, is_training=False)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.train_config.batch_size, 
            shuffle=True,
            num_workers=self.train_config.num_workers, 
            collate_fn=collate_fn, 
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.train_config.batch_size * 2,
            shuffle=False,
            num_workers=self.train_config.num_workers, 
            collate_fn=collate_fn, 
            pin_memory=True
        )
        
        logger.info(f"📈 Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    def train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Use cosine scheduler for some epochs
        if epoch % 10 < 5:  # Use cosine for first half of every 10 epochs
            current_scheduler = self.cosine_scheduler
        else:
            current_scheduler = None
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            images = batch['image'].to(self.device, non_blocking=True)
            prices = batch['price'].to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast(enabled=self.train_config.use_amp):
                predictions = self.model(input_ids, attention_mask, images)
                loss = self.criterion(predictions, prices)
                loss = loss / self.train_config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if current_scheduler:
                    current_scheduler.step(epoch + batch_idx / num_batches)
            
            total_loss += loss.item() * self.train_config.gradient_accumulation_steps
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                images = batch['image'].to(self.device, non_blocking=True)
                prices = batch['price'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.train_config.use_amp):
                    predictions = self.model(input_ids, attention_mask, images)
                    loss = self.criterion(predictions, prices)
                
                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(prices.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate SMAPE
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        smape_score = calculate_smape_metric(y_true, y_pred)
        
        return avg_loss, smape_score
    
    def save_checkpoint(self, epoch: int, smape: float, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_smape': smape,
            'config': self.train_config.__dict__,
            'history': self.history
        }
        
        if is_best:
            save_path = "src/model_checkpoints/continued2_best_model.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"💾 New best model saved: {save_path} (SMAPE: {smape:.2f}%)")
    
    def continue_training(self, train_csv_path: str):
        """Main continued training loop"""
        logger.info("🚀 Starting continued training...")
        logger.info("=" * 60)
        
        # Load checkpoint and setup
        self.load_checkpoint()
        self.setup_optimization()
        self.prepare_data(train_csv_path)
        
        # Training loop
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(1, self.train_config.num_epochs + 1):
            logger.info(f"\n{'='*20} Continued Epoch {epoch}/{self.train_config.num_epochs} {'='*20}")
            
            # Train and validate
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_smape = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_smape'].append(val_smape)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.2f}%")
            
            # Check for improvement
            is_best = val_smape < self.best_smape
            if is_best:
                self.best_smape = val_smape
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_smape, is_best=True)
                
                # Check if target achieved
                if val_smape < self.target_smape:
                    logger.info(f"🎉 TARGET ACHIEVED! SMAPE < {self.target_smape}%")
                    logger.info(f"🏆 Final SMAPE: {val_smape:.2f}%")
                    break
            else:
                epochs_without_improvement += 1
            
            # Learning rate scheduling
            self.scheduler.step(val_smape)
            
            # Early stopping check
            if epochs_without_improvement >= self.train_config.patience:
                logger.info(f"⏹️ Early stopping triggered after {epochs_without_improvement} epochs")
                break
            
            # Memory cleanup
            memory_cleanup()
        
        # Training complete
        total_time = time.time() - start_time
        hours, minutes = int(total_time // 3600), int((total_time % 3600) // 60)
        
        logger.info("=" * 60)
        logger.info("🏁 Continued training completed!")
        logger.info(f"⏱️ Total time: {hours}h {minutes}m")
        logger.info(f"🎯 Best SMAPE achieved: {self.best_smape:.2f}%")
        
        if self.best_smape < self.target_smape:
            logger.info(f"✅ SUCCESS: Target SMAPE < {self.target_smape}% achieved!")
        else:
            logger.info(f"📈 Progress made, consider additional training")
        
        return self.best_smape

def main():
    """Main execution function"""
    logger.info("🔄 ML Challenge 2025 - Continued Training")
    logger.info("🎯 Goal: Improve SMAPE to < 40%")
    logger.info("=" * 60)
    
    # Configuration
    checkpoint_path = "src/model_checkpoints/continued_best_model.pt"
    train_csv = "dataset/train.csv"
    target_smape = 40.0
    
    # Check files exist
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.info("💡 Please train the initial model first")
        return
    
    if not os.path.exists(train_csv):
        logger.error(f"❌ Training data not found: {train_csv}")
        return
    
    try:
        # Create continued trainer
        trainer = ContinuedTrainer(checkpoint_path, target_smape)
        
        # Start continued training
        final_smape = trainer.continue_training(train_csv)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("📊 FINAL RESULTS")
        logger.info(f"🎯 Target SMAPE: < {target_smape}%")
        logger.info(f"🏆 Achieved SMAPE: {final_smape:.2f}%")
        
        if final_smape < target_smape:
            logger.info("🎉 SUCCESS! Target achieved!")
            logger.info("📁 Best model saved as: continued_best_model.pt")
        else:
            improvement = 55.82 - final_smape  # Assuming previous best was 55.82%
            logger.info(f"📈 Improvement: {improvement:.2f} percentage points")
            logger.info("💡 Consider running additional training cycles")
        
    except Exception as e:
        logger.error(f"❌ Continued training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()