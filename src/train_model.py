# ===================================================================
# 1. IMPORTS
# ===================================================================
import gc
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Field.*has no effect.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# (Assuming these are your custom modules)
from data_loader import (ProductDataset, TextProcessor, ImageProcessor,
                         collate_fn, create_data_loaders)
from multimodal_model import MultimodalPricePredictionModel, create_model

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 2. CONFIGURATION & LOGGING
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for the model and training process."""
    # Model Architecture
    text_model_name: str = "distilbert-base-uncased"
    image_model_name: str = "efficientnet_b0"
    embedding_dim: int = 768
    dropout_rate: float = 0.2

    # Data & Preprocessing
    image_size: int = 224
    max_text_length: int = 128

    # Training Hyperparameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2  # Effective batch size = batch_size * this
    patience: int = 3
    min_delta: float = 0.001
    warmup_ratio: float = 0.1  # Warmup ratio for learning rate scheduling

    # System & Performance
    num_workers: int = os.cpu_count() // 2
    use_amp: bool = True  # Automatic Mixed Precision

# ===================================================================
# 3. CUSTOM LOSS FUNCTIONS
# ===================================================================
class SMAPELoss(nn.Module):
    """
    Implements the Symmetric Mean Absolute Percentage Error (SMAPE) loss.
    SMAPE is bounded between 0% and 200%.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the SMAPE loss."""
        numerator = torch.abs(predictions - targets)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0
        
        # Add epsilon to prevent division by zero
        loss = numerator / (denominator + self.epsilon)
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    """
    A combined loss function that blends SMAPE and MSE for training stability.
    This helps guide the model with a smoother MSE loss while still optimizing
    for the primary SMAPE metric.
    """
    def __init__(self, smape_weight: float = 0.7, mse_weight: float = 0.3):
        super().__init__()
        self.smape_loss = SMAPELoss()
        self.mse_loss = nn.MSELoss()
        self.smape_weight = smape_weight
        self.mse_weight = mse_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the combined weighted loss."""
        smape = self.smape_loss(predictions, targets)
        mse = self.mse_loss(predictions, targets)
        
        # Combining the two losses with their respective weights
        combined_loss = (self.smape_weight * smape) + (self.mse_weight * mse)
        return combined_loss

# ===================================================================
# 4. UTILITY CLASS & FUNCTIONS
# ===================================================================
class EarlyStopping:
    """
    Utility to stop training when a monitored metric has stopped improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Checks if training should be stopped.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def calculate_smape_metric(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculates the SMAPE metric score in percentage."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    score = np.mean(numerator / (denominator + epsilon))
    return score * 100

def memory_cleanup():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===================================================================
# 5. THE TRAINER CLASS
# ===================================================================
class Trainer:
    """
    The main class to orchestrate the training and validation pipeline.
    """
    def __init__(self, config: ModelConfig, save_dir: str = "model_checkpoints"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Model, Loss, Optimizer
        self.model = create_model(config).to(self.device)
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scaler = GradScaler(enabled=config.use_amp)
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # History tracking
        self.history = {'train_loss': [], 'val_loss': [], 'val_smape': []}

    def _prepare_data(self, train_csv_path: str, val_split: float = 0.15) -> None:
        """
        Prepares training and validation DataLoaders.
        """
        LOGGER.info("Preparing data loaders...")
        df = pd.read_csv(train_csv_path)

        # Stratified split to ensure price distribution is similar in train/val
        splitter = train_test_split(
            df,
            test_size=val_split,
            random_state=42,
            stratify=pd.cut(np.log1p(df['price']), bins=10, labels=False, duplicates='drop')
        )
        train_df, val_df = splitter
        train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
        
        # Assuming TextProcessor and ImageProcessor are defined in data_loader.py
        from data_loader import ImageCache
        
        text_processor = TextProcessor(self.config.text_model_name, self.config.max_text_length)
        train_image_proc = ImageProcessor(self.config.image_size, is_training=True)
        val_image_proc = ImageProcessor(self.config.image_size, is_training=False)
        image_cache = ImageCache()  # Create image cache for downloading/caching images

        # Save DataFrames to CSV files
        train_csv_path = self.save_dir / "train_split.csv"
        val_csv_path = self.save_dir / "val_split.csv"
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        train_dataset = ProductDataset(str(train_csv_path), text_processor, train_image_proc, image_cache, is_training=True)
        val_dataset = ProductDataset(str(val_csv_path), text_processor, val_image_proc, image_cache, is_training=False)
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size * 2, # Larger batch for validation
            num_workers=self.config.num_workers, collate_fn=collate_fn, pin_memory=True
        )

        LOGGER.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} validation samples.")
        
        # Setup scheduler after data loaders are created
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=len(self.train_loader) * self.config.num_epochs,
            pct_start=self.config.warmup_ratio
        )

    def _train_one_epoch(self) -> float:
        """Performs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            images = batch['image'].to(self.device, non_blocking=True)
            prices = batch['price'].to(self.device, non_blocking=True)

            with autocast(enabled=self.config.use_amp):
                predictions = self.model(input_ids, attention_mask, images)
                loss = self.criterion(predictions, prices)
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step (occurs every accumulation_steps)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix(
                loss=f'{total_loss / (batch_idx + 1):.4f}',
                lr=f'{self.scheduler.get_last_lr()[0]:.2e}'
            )

        return total_loss / len(self.train_loader)

    def _validate(self) -> Tuple[float, float]:
        """Performs a single validation run."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                images = batch['image'].to(self.device, non_blocking=True)
                prices = batch['price'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config.use_amp):
                    predictions = self.model(input_ids, attention_mask, images)
                    loss = self.criterion(predictions, prices)
                
                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(prices.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate SMAPE score
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        smape_score = calculate_smape_metric(y_true, y_pred)
        
        return avg_loss, smape_score

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves a model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        if is_best:
            filepath = self.save_dir / "best_model.pt"
            torch.save(state, filepath)
            LOGGER.info(f"Saved new best model to {filepath}")

    def _plot_and_save_history(self):
        """Plots and saves the training history curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)

        ax1.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, self.history['val_smape'], 'go-', label='Validation SMAPE')
        ax2.set_title('Validation SMAPE (%)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('SMAPE (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = self.save_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        LOGGER.info(f"Training curves saved to {save_path}")

    def run_training(self, train_csv_path: str):
        """The main training loop."""
        self._prepare_data(train_csv_path)
        best_val_smape = float('inf')

        for epoch in range(1, self.config.num_epochs + 1):
            LOGGER.info(f"\n{'='*20} Epoch {epoch}/{self.config.num_epochs} {'='*20}")
            
            train_loss = self._train_one_epoch()
            val_loss, val_smape = self._validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_smape'].append(val_smape)

            LOGGER.info(
                f"Epoch {epoch} Summary: "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val SMAPE: {val_smape:.2f}%"
            )

            is_best = val_smape < best_val_smape
            if is_best:
                best_val_smape = val_smape
                self._save_checkpoint(epoch, is_best=True)
            
            if self.early_stopping(val_loss, self.model):
                LOGGER.info(f"Early stopping triggered at epoch {epoch}.")
                if self.early_stopping.restore_best_weights:
                    self.model.load_state_dict(self.early_stopping.best_weights)
                    LOGGER.info("Restored best model weights.")
                break

            memory_cleanup()

        LOGGER.info(f"\nTraining finished. Best Validation SMAPE: {best_val_smape:.2f}%")
        self._plot_and_save_history()
        
        return best_val_smape

# ===================================================================
# 6. MAIN EXECUTION
# ===================================================================
def main():
    """Main function to configure and run the training process."""
    # --- Configuration ---
    # Using the default config, but you can override it here.
    # Especially useful for GPUs with less VRAM.
    config = ModelConfig(
        batch_size=16,
        gradient_accumulation_steps=4, # Effective batch size = 16*4 = 64
        num_epochs=12,
        learning_rate=2e-5
    )

    # --- Initialization & Training ---
    trainer = Trainer(config, save_dir="multimodal_price_model")
    
    # Path to your training data
    train_csv_path = "../dataset/train.csv" # Adjust path as needed
    
    if not Path(train_csv_path).exists():
        LOGGER.error(f"Training file not found at: {train_csv_path}")
        return
        
    trainer.run_training(train_csv_path)


if __name__ == "__main__":
    main()