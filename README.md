# 🛍️ Smart Product Pricing Challenge - Multimodal ML Solution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated multimodal deep learning solution for predicting e-commerce product prices using both textual catalog content and product images. This project implements a state-of-the-art transformer-based architecture with cross-attention mechanisms to achieve optimal pricing predictions.

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Technical Implementation](#-technical-implementation)
- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)

## 🎯 Problem Statement

### Challenge Overview
In e-commerce, determining optimal product pricing is crucial for marketplace success and customer satisfaction. The challenge requires developing an ML solution that analyzes multimodal product data to predict accurate prices.

### Key Constraints
- **Dataset**: 75,000 training samples, 75,000 test samples
- **Evaluation Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Model Limit**: ≤8 Billion parameters
- **License**: MIT/Apache 2.0 compliant models only
- **Strict Policy**: No external price lookup allowed

### Data Structure
```
📊 Input Features:
├── sample_id: Unique identifier
├── catalog_content: Product titles, descriptions, specifications
├── image_link: Product image URLs
└── price: Target variable (training only)

🎯 Output Format:
├── sample_id: Matching test sample IDs
└── price: Predicted positive float values
```

### Evaluation Formula
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```
- **Range**: 0% to 200% (lower is better)
- **Example**: Actual=$100, Predicted=$120 → SMAPE=18.18%

## 🏗️ Solution Architecture

### Multimodal Deep Learning Approach

Our solution implements a sophisticated **cross-attention multimodal architecture** that effectively combines textual and visual features for price prediction.

```
🧠 Model Architecture:
├── Text Encoder (DistilBERT)
│   ├── Advanced text preprocessing
│   ├── Structured information extraction
│   └── 768-dim embeddings
├── Image Encoder (EfficientNet-B0)
│   ├── Robust image processing
│   ├── Data augmentation pipeline
│   └── 512-dim visual features
├── Cross-Attention Mechanism
│   ├── Multi-head attention (8 heads)
│   ├── Text-to-image attention
│   └── Feature fusion
└── Price Prediction Head
    ├── Multi-layer regression
    ├── Softplus activation
    └── Positive price guarantee
```

## 🔧 Technical Implementation

### Core Components

#### 1. **Text Processing Pipeline** (`data_loader.py`)
- **Advanced Text Cleaning**: Regex-based structured information extraction
- **Feature Engineering**: Item names, bullet points, specifications parsing
- **Tokenization**: DistilBERT tokenizer with 256 max length
- **Memory Optimization**: Efficient text length management

#### 2. **Image Processing Pipeline** (`data_loader.py`)
- **Robust Download System**: Retry logic with caching mechanism
- **Data Augmentation**: Albumentations-based transformations
- **Preprocessing**: ImageNet normalization and resizing
- **Cache Management**: Persistent image storage for efficiency

#### 3. **Multimodal Architecture** (`multimodal_model.py`)
```python
class MultimodalPricePredictionModel(nn.Module):
    - TextEncoder: DistilBERT + projection layers
    - ImageEncoder: EfficientNet + feature extraction
    - CrossAttention: Multi-head attention mechanism
    - FusionLayers: Feature combination and processing
    - PricePredictor: Regression head with positive constraints
```

#### 4. **Training Framework** (`train_model.py`)
- **Custom Loss Functions**: 
  - `SMAPELoss`: Direct SMAPE optimization
  - `CombinedLoss`: Weighted SMAPE + MSE for stability
- **Advanced Optimization**:
  - AdamW optimizer with weight decay
  - OneCycleLR scheduling
  - Gradient accumulation for effective larger batches
- **Training Enhancements**:
  - Mixed precision training (AMP)
  - Early stopping with patience
  - Memory management and GPU optimization

#### 5. **Prediction Pipeline** (`predict.py`)
- **Single Model Inference**: Optimized batch processing
- **Ensemble Methods**: Weighted combination of multiple models
- **Output Validation**: Format compliance and constraint checking
- **Memory Efficiency**: GPU cache management during inference

### Key Innovation Features

#### 🎯 **Cross-Attention Mechanism**
```python
def forward(self, text_features, image_features):
    # Multi-head attention between text and image
    Q = self.text_to_q(text_features)  # Query from text
    K = self.image_to_k(image_features)  # Key from image
    V = self.image_to_v(image_features)  # Value from image
    
    attention_output = self.multi_head_attention(Q, K, V)
    return self.layer_norm(text_features + attention_output)
```

#### 📊 **Advanced Text Feature Engineering**
- Structured parsing of product specifications
- Bullet point extraction and prioritization
- Value-unit pair recognition
- Brand and category identification

#### 🖼️ **Robust Image Processing**
- Intelligent retry mechanisms for image downloads
- Comprehensive data augmentation strategies
- EfficientNet-based feature extraction
- Fallback handling for failed downloads

## 💻 System Requirements

### Development Environment
```
🖥️ Hardware Specifications:
├── CPU: Intel Core i5-12450HX
├── RAM: 16GB DDR5
├── GPU: NVIDIA RTX 3050 (6GB VRAM)
└── Storage: 512 GB SSD

⚙️ Software Requirements:
├── Python: 3.8 - 3.11
├── CUDA: 11.8+ (for GPU acceleration)
├── Operating System: Windows 10
└── Available Disk Space: 40 GB+ (for datasets and models)
```

### Performance Optimizations
- **Memory Efficient**: Optimized for 6GB VRAM GPUs
- **Batch Processing**: Dynamic batch size adjustment
- **Mixed Precision**: FP16 training for 2x speed improvement
- **Gradient Accumulation**: Effective larger batch sizes without memory increase

## 🚀 Installation & Setup

### Step 1: Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd smart-product-pricing

# Create virtual environment
python -m venv .venv
On Windows: .venv\Scripts\activate

# Verify CUDA installation (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Dataset Preparation
```
📁 Required Dataset Structure:
dataset/
├── train.csv          # 75k training samples
├── test.csv           # 75k test samples  
├── sample_test.csv    # Sample test file
└── sample_test_out.csv # Expected output format
```

## 📖 Usage Guide

### Training the Model

#### Quick Start Training
```bash
# Run complete training pipeline
python src/run_pipeline.py
```

#### Custom Training Configuration
```python
# Modify training parameters in train_model.py
config = ModelConfig()
config.batch_size = 16          # Adjust for your GPU
config.learning_rate = 2e-5     # Fine-tune learning rate
config.num_epochs = 10          # Training duration
config.hidden_dim = 512         # Model complexity
```

### Making Predictions

#### Generate Test Predictions
```bash
# Use best trained model
python src/predict.py
```

#### Custom Prediction Script
```python
from src.predict import ModelPredictor

# Load trained model
predictor = ModelPredictor("src/model_checkpoints/best_model.pt")

# Predict on test set
results = predictor.predict_dataset("dataset/test.csv")
results.to_csv("predictions.csv", index=False)
```

### Model Evaluation
```python
# Calculate SMAPE score
from src.train_model import calculate_smape_metric

smape_score = calculate_smape_metric(y_true, y_pred)
print(f"SMAPE Score: {smape_score:.2f}%")
```

## 📁 Project Structure

```
smart-product-pricing/
├── 📄 README.md                    # Project documentation
├── 📂 dataset/                     # Training and test data
│   ├── train.csv                   # Training dataset (75k samples)
│   ├── test.csv                    # Test dataset (75k samples)
│   ├── sample_test.csv             # Sample test data
│   └── sample_test_out.csv         # Expected output format
├── 📂 src/                         # Source code
│   ├── 🧠 multimodal_model.py      # Core model architecture
│   ├── 📊 data_loader.py           # Data processing pipeline
│   ├── 🏋️ train_model.py           # Training framework
│   ├── 🔮 predict.py               # Prediction pipeline
│   ├── 🔄 continue_training.py     # Advanced training
│   ├── 🚀 run_pipeline.py          # Complete workflow
│   ├── 🧪 test_setup.py            # Environment verification
│   └── 📂 model_checkpoints/       # Trained models
│       ├── best_model.pt
│       ├── continued_best_model.pt
│       ├── training_curves.png
│       ├── train_split.csv
│       └── val_split.csv
```

## 📚 Dependencies

### Core ML Libraries
```python
# Deep Learning Framework
torch==2.0.1+cu118              # PyTorch with CUDA support
torchvision==0.15.2+cu118       # Computer vision utilities
torch-audio==2.0.2+cu118        # Audio processing (dependency)

# Transformer Models
transformers==4.30.2            # Hugging Face transformers
tokenizers==0.13.3              # Fast tokenization

# Computer Vision
timm==0.9.2                     # PyTorch Image Models
albumentations==1.3.1           # Advanced image augmentations
opencv-python==4.8.0.74         # Computer vision operations

# Data Processing
pandas==2.0.3                   # Data manipulation
numpy==1.24.3                   # Numerical computing
scikit-learn==1.3.0             # Machine learning utilities
```

### Utility Libraries
```python
# Image Processing
Pillow==10.0.0                  # Python Imaging Library
requests==2.31.0                # HTTP library for downloads

# Visualization & Logging
matplotlib==3.7.1               # Plotting library
seaborn==0.12.2                 # Statistical visualization
tqdm==4.65.0                    # Progress bars
logging                         # Built-in logging (Python std)

# System & Performance
psutil==5.9.5                   # System monitoring
memory-profiler==0.61.0         # Memory usage tracking
```

### Development Tools
```python
# Code Quality
black==23.7.0                   # Code formatting
flake8==6.0.0                   # Linting
pytest==7.4.0                   # Testing framework

# Jupyter Notebook Support
jupyter==1.0.0                  # Jupyter ecosystem
ipykernel==6.25.0              # Jupyter kernel
notebook==7.0.0                 # Jupyter notebook interface
```

## 🏆 Key Achievements

### Technical Innovations
✅ **Multimodal Architecture**: Successfully combined text and image features using cross-attention  
✅ **Memory Optimization**: Efficient training on consumer-grade GPU (RTX 3050)  
✅ **Robust Data Pipeline**: Handles image download failures and text processing edge cases  
✅ **Production Ready**: Comprehensive error handling, logging, and validation  
✅ **Ensemble Methods**: Multiple model combination for improved performance  

### Performance Highlights
🎯 **SMAPE Score**: Achieved 48% on validation set  
⚡ **Training Speed**: Optimized for 6GB VRAM with mixed precision  
🔄 **Scalability**: Handles 75k sample datasets efficiently  
💾 **Memory Efficient**: Smart caching and garbage collection  
## 🙏 Acknowledgments

- **Hugging Face**: For the excellent transformers library
- **PyTorch Team**: For the deep learning framework
- **EfficientNet Authors**: For the efficient computer vision architecture
- **Competition Organizers**: For providing the challenging dataset and problem statement

---