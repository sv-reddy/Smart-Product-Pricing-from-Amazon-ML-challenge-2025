import os
import torch
import torch.nn as nn
import warnings
from torch.amp import autocast
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Optional
import gc

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Field.*has no effect.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from multimodal_model import create_model, ModelConfig
from data_loader import create_data_loaders, TextProcessor, ImageCache, ImageProcessor, ProductDataset, collate_fn
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPredictor:
    """Class for making predictions with the trained multimodal model"""
    
    def __init__(self, model_checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the predictor with a trained model checkpoint
        
        Args:
            model_checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device, weights_only=False     )
        
        # Create model from config
        config_dict = checkpoint['config']
        self.config = ModelConfig()
        for key, value in config_dict.items():
            setattr(self.config, key, value)
        
        # Initialize model
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully from {model_checkpoint_path}")
        val_smape = checkpoint.get('val_smape', 'N/A')
        if isinstance(val_smape, (int, float)):
            logger.info(f"Model validation SMAPE: {val_smape:.4f}%")
        else:
            logger.info(f"Model validation SMAPE: {val_smape}")
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.image_cache = ImageCache()
        self.image_processor = ImageProcessor(
            image_size=self.config.image_size, 
            is_training=False
        )
        
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Make predictions on a batch of data"""
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            images = batch['image'].to(self.device, non_blocking=True)
            
            if self.config.use_amp:
                with autocast('cuda'):
                    predictions = self.model(input_ids, attention_mask, images)
            else:
                predictions = self.model(input_ids, attention_mask, images)
            
            return predictions.cpu().numpy()
    
    def predict_dataset(self, csv_path: str, batch_size: int = 32, num_workers: int = 4) -> pd.DataFrame:
        
        logger.info(f"Starting prediction on {csv_path}")
        
        # Create dataset
        dataset = ProductDataset(
            csv_path=csv_path,
            text_processor=self.text_processor,
            image_processor=self.image_processor,
            image_cache=self.image_cache,
            is_training=False
        )
        
        # Create data loader with optimized settings
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
            prefetch_factor=2  # Prefetch batches for better pipeline
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples, {len(data_loader)} batches")
        
        # Make predictions
        all_sample_ids = []
        all_predictions = []
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Predicting")):
            # Get sample IDs
            sample_ids = batch['sample_id'].numpy()
            all_sample_ids.extend(sample_ids)
            
            # Make predictions
            predictions = self.predict_batch(batch)
            all_predictions.extend(predictions)
            
            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'sample_id': all_sample_ids,
            'price': all_predictions
        })
        
        # Ensure positive prices
        results_df['price'] = np.maximum(results_df['price'], 0.01)
        
        logger.info(f"Predictions completed: {len(results_df)} samples")
        logger.info(f"Price range: ${results_df['price'].min():.2f} - ${results_df['price'].max():.2f}")
        logger.info(f"Mean price: ${results_df['price'].mean():.2f}")
        
        return results_df
    
    def predict_single(self, catalog_content: str, image_url: str) -> float:
        """
        Make a prediction on a single product
        
        Args:
            catalog_content: Product description text
            image_url: URL to product image
        
        Returns:
            Predicted price
        """
        # Process text
        text_data = self.text_processor.tokenize(catalog_content, max_length=self.config.max_text_length)
        
        # Process image
        image = self.image_cache.get_image(image_url)
        image_tensor = self.image_processor.process_image(image)
        
        # Create batch tensors
        input_ids = text_data['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = text_data['attention_mask'].unsqueeze(0).to(self.device)
        images = image_tensor.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if self.config.use_amp:
                with autocast('cuda'):
                    prediction = self.model(input_ids, attention_mask, images)
            else:
                prediction = self.model(input_ids, attention_mask, images)
        
        return float(prediction.cpu().numpy()[0])

class EnsemblePredictor:
    """Ensemble predictor that combines multiple models"""
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """
        Initialize ensemble predictor
        
        Args:
            model_paths: List of paths to model checkpoints
            weights: Weights for each model (if None, equal weights are used)
        """
        self.predictors = [ModelPredictor(path) for path in model_paths]
        
        if weights is None:
            self.weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            assert len(weights) == len(model_paths), "Number of weights must match number of models"
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        logger.info(f"Initialized ensemble with {len(self.predictors)} models")
        logger.info(f"Model weights: {self.weights}")
    
    def predict_dataset(self, csv_path: str, batch_size: int = 16) -> pd.DataFrame:
        """Make ensemble predictions on dataset"""
        logger.info("Making ensemble predictions...")
        
        predictions_list = []
        for i, predictor in enumerate(self.predictors):
            logger.info(f"Getting predictions from model {i+1}/{len(self.predictors)}")
            pred_df = predictor.predict_dataset(csv_path, batch_size)
            predictions_list.append(pred_df['price'].values)
        
        # Combine predictions using weighted average
        sample_ids = pred_df['sample_id'].values
        ensemble_predictions = np.zeros(len(sample_ids))
        
        for predictions, weight in zip(predictions_list, self.weights):
            ensemble_predictions += predictions * weight
        
        # Create final DataFrame
        results_df = pd.DataFrame({
            'sample_id': sample_ids,
            'price': ensemble_predictions
        })
        
        logger.info(f"Ensemble predictions completed: {len(results_df)} samples")
        logger.info(f"Final price range: ${results_df['price'].min():.2f} - ${results_df['price'].max():.2f}")
        
        return results_df

def generate_test_predictions(
    model_checkpoint: str,
    test_csv: str,
    output_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_ensemble: bool = False,
    ensemble_models: Optional[List[str]] = None
) -> None:
    
    logger.info("=" * 50)
    logger.info("GENERATING TEST PREDICTIONS")
    logger.info("=" * 50)
    
    try:
        if use_ensemble and ensemble_models:
            # Use ensemble prediction
            predictor = EnsemblePredictor(ensemble_models)
            predictions_df = predictor.predict_dataset(test_csv, batch_size)
        else:
            # Use single model prediction
            predictor = ModelPredictor(model_checkpoint)
            predictions_df = predictor.predict_dataset(test_csv, batch_size, num_workers)
        
        # Save predictions
        predictions_df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to: {output_csv}")
        
        # Validate output format
        validate_output_format(output_csv, test_csv)
        
        logger.info("Prediction generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during prediction generation: {e}")
        raise

def validate_output_format(output_csv: str, test_csv: str) -> None:
    """Validate that the output format matches requirements"""
    logger.info("Validating output format...")
    
    # Load files
    output_df = pd.read_csv(output_csv)
    test_df = pd.read_csv(test_csv)
    
    # Check columns
    required_columns = ['sample_id', 'price']
    assert all(col in output_df.columns for col in required_columns), f"Missing columns. Required: {required_columns}"
    
    # Check sample count
    assert len(output_df) == len(test_df), f"Sample count mismatch: {len(output_df)} vs {len(test_df)}"
    
    # Check sample IDs match
    output_ids = set(output_df['sample_id'])
    test_ids = set(test_df['sample_id'])
    assert output_ids == test_ids, "Sample IDs don't match between test and output files"
    
    # Check for positive prices
    assert all(output_df['price'] > 0), "All prices must be positive"
    
    # Check for NaN values
    assert not output_df['price'].isna().any(), "No NaN prices allowed"
    
    logger.info("✅ Output format validation passed!")
    logger.info(f"Samples: {len(output_df)}")
    logger.info(f"Price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")

def main():
    """Main function for generating predictions"""
    # Configuration
    model_checkpoint = "src/model_checkpoints/aggressive_best_model.pt"
    test_csv = "dataset/test.csv"
    output_csv = "dataset/test_out.csv"
    
    # Check if model exists
    if not os.path.exists(model_checkpoint):
        logger.error(f"Model checkpoint not found: {model_checkpoint}")
        logger.error("Please train the model first using train_model.py")
        return
    
    # Generate predictions
    generate_test_predictions(
        model_checkpoint=model_checkpoint,
        test_csv=test_csv,
        output_csv=output_csv,
        batch_size=32,  # Increased from 16 for faster inference
        num_workers=4   # Increased from 2 for faster data loading
    )

if __name__ == "__main__":
    main()