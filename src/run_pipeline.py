import os
import sys
import torch
import logging
import time
import warnings
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Field.*has no effect.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from multimodal_model import create_model, ModelConfig
from train_model import Trainer
from predict import generate_test_predictions, validate_output_format

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if system meets requirements"""
    logger.info("=" * 60)
    logger.info("SYSTEM REQUIREMENTS CHECK")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {device_name}")
        logger.info(f"✅ VRAM: {vram_gb:.1f} GB")
        
        if vram_gb < 4:
            logger.warning("⚠️ Low VRAM detected. Consider reducing batch size.")
    else:
        logger.warning("⚠️ CUDA not available. Training will be very slow on CPU.")
    
    # Check PyTorch version
    logger.info(f"✅ PyTorch version: {torch.__version__}")
    
    # Check data files
    dataset_dir = Path("dataset")  # Changed from ../dataset since we run from parent dir
    required_files = ["train.csv", "test.csv", "sample_test.csv", "sample_test_out.csv"]
    
    for file in required_files:
        file_path = dataset_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1e6
            logger.info(f"✅ {file}: {size_mb:.1f} MB")
        else:
            logger.error(f"❌ Missing file: {file_path}")
            return False
    
    logger.info("System requirements check completed!")
    return True

def test_model_architecture():
    """Test model creation and forward pass"""
    logger.info("=" * 60)
    logger.info("TESTING MODEL ARCHITECTURE")
    logger.info("=" * 60)
    
    try:
        # Create model config optimized for RTX 3050
        config = ModelConfig()
        config.batch_size = 8  # Conservative for 6GB VRAM
        config.hidden_dim = 512
        config.max_text_length = 256
        config.image_size = 224
        
        # Create model
        model = create_model(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model created successfully!")
        logger.info(f"✅ Total parameters: {total_params:,}")
        logger.info(f"✅ Trainable parameters: {trainable_params:,}")
        logger.info(f"✅ Model size: ~{total_params * 4 / 1e6:.1f} MB (FP32)")
        
        # Test forward pass
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, config.max_text_length)).to(device)
        attention_mask = torch.ones(batch_size, config.max_text_length).to(device)
        images = torch.randn(batch_size, 3, config.image_size, config.image_size).to(device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids, attention_mask, images)
            forward_time = time.time() - start_time
        
        logger.info(f"✅ Forward pass successful!")
        logger.info(f"✅ Output shape: {outputs.shape}")
        logger.info(f"✅ Forward pass time: {forward_time:.3f} seconds")
        logger.info(f"✅ Sample predictions: {outputs.cpu().numpy()}")
        
        # Clean up memory
        del model, input_ids, attention_mask, images, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model architecture test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model architecture test failed: {e}")
        return False

def run_training():
    """Run the training pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Enhanced configuration for better performance (targeting <25% SMAPE)
        config = ModelConfig()
        config.batch_size = 12         # Increased from 8 - your GPU can handle it
        config.gradient_accumulation_steps = 6  # Effective batch size = 72 (increased)
        config.num_epochs = 10        # Significantly increased for better convergence
        config.learning_rate = 2e-5    # Increased from 1e-5 for faster learning
        config.use_amp = True          # Keep mixed precision for memory efficiency
        config.num_workers = 3         # Increased as you suggested - good with your CPU
        config.weight_decay = 1e-4     # Add regularization
        config.warmup_ratio = 0.1      # Add learning rate warmup
        config.patience = 5            # Increased patience for longer training
        
        # Create trainer
        trainer = Trainer(config, save_dir="src/model_checkpoints")  # Fixed path
        
        # Start training
        train_csv = "dataset/train.csv"  # Changed from ../dataset/train.csv
        best_smape = trainer.run_training(train_csv)
        
        logger.info(f"Training completed with best SMAPE: {best_smape:.4f}%")
        
        # Performance evaluation with enhanced targets
        if best_smape < 20:
            logger.info("🎉 EXCELLENT: SMAPE < 20% - Outstanding performance!")
        elif best_smape < 25:
            logger.info("🎯 TARGET ACHIEVED: SMAPE < 25% - Excellent performance!")
        elif best_smape < 30:
            logger.info("👍 GOOD: SMAPE < 30% - Strong performance!")
        elif best_smape < 40:
            logger.info("📈 FAIR: SMAPE < 40% - Reasonable performance, room for improvement")
        else:
            logger.info("⚠️ NEEDS IMPROVEMENT: SMAPE > 40% - Consider hyperparameter tuning")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def run_prediction():
    """Generate predictions for test dataset"""
    logger.info("=" * 60)
    logger.info("GENERATING TEST PREDICTIONS")
    logger.info("=" * 60)
    
    try:
        model_checkpoint = "src/model_checkpoints/best_model.pt"  # Fixed path
        test_csv = "dataset/test.csv"  # Fixed path
        output_csv = "dataset/test_out.csv"  # This will be the submission file
        
        # Check if model exists
        if not os.path.exists(model_checkpoint):
            logger.error(f"❌ Model checkpoint not found: {model_checkpoint}")
            return False
        
        # Generate predictions
        generate_test_predictions(
            model_checkpoint=model_checkpoint,
            test_csv=test_csv,
            output_csv=output_csv,
            batch_size=16  # Can be higher for inference
        )
        
        logger.info("✅ Test predictions generated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction generation failed: {e}")
        return False

def main():
    """Main execution function"""
    start_time = time.time()
    
    logger.info("🚀 STARTING MULTIMODAL DEEP LEARNING PIPELINE")
    logger.info("🎯 Target: Smart Product Pricing with 80%+ accuracy")
    logger.info("💻 Optimized for: RTX 3050 (6GB VRAM) + 16GB RAM")
    logger.info("=" * 60)
    
    # Step 1: Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met. Please check your setup.")
        return
    
    # Step 2: Test model architecture
    if not test_model_architecture():
        logger.error("❌ Model architecture test failed. Please check the code.")
        return
    
    # Step 3: Run training
    if not run_training():
        logger.error("❌ Training failed. Please check the logs.")
        return
    
    # Step 4: Generate predictions
    if not run_prediction():
        logger.error("❌ Prediction generation failed. Please check the logs.")
        return
    
    # Success!
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    logger.info("=" * 60)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"⏱️ Total time: {hours}h {minutes}m")
    logger.info(f"📊 Check model_checkpoints/ for training results")
    logger.info(f"📁 Predictions saved to: dataset/test_out.csv")
    logger.info("=" * 60)
    
    # Final validation
    try:
        validate_output_format("dataset/test_out.csv", "dataset/test.csv")
        logger.info("✅ Output format validation passed!")
    except Exception as e:
        logger.error(f"❌ Output validation failed: {e}")

if __name__ == "__main__":
    main()