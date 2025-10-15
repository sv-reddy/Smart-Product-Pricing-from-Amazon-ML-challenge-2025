import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import AutoModel, AutoTokenizer
import timm
from typing import Dict, List, Tuple, Optional
import math

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Field.*has no effect.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class MultiHeadCrossAttention(nn.Module):
    """Cross-attention mechanism between text and image features"""
    
    def __init__(self, text_dim: int, image_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.head_dim = text_dim // num_heads
        
        assert text_dim % num_heads == 0, "text_dim must be divisible by num_heads"
        
        # Linear projections for query, key, value
        self.text_to_q = nn.Linear(text_dim, text_dim)
        self.image_to_k = nn.Linear(image_dim, text_dim)
        self.image_to_v = nn.Linear(image_dim, text_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(text_dim)
        
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        batch_size = text_features.size(0)
        
        # Project to Q, K, V
        Q = self.text_to_q(text_features)  # [batch, text_dim]
        K = self.image_to_k(image_features)  # [batch, text_dim]
        V = self.image_to_v(image_features)  # [batch, text_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # [batch, heads, head_dim]
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch, heads, head_dim]
        attended = attended.view(batch_size, self.text_dim)  # [batch, text_dim]
        
        # Residual connection and layer norm
        output = self.layer_norm(text_features + attended)
        return output

class TextEncoder(nn.Module):
    """DistilBERT-based text encoder for catalog content"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # Freeze early layers to save memory and computation
        for param in list(self.bert.parameters())[:30]:  # Freeze first 30 parameters
            param.requires_grad = False
            
        # Project BERT output to desired hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, bert_dim]
        
        # Project to desired dimension
        text_features = self.projection(cls_output)  # [batch, hidden_dim]
        return text_features

class ImageEncoder(nn.Module):
    """EfficientNet-based image encoder for product images"""
    
    def __init__(self, model_name: str = "efficientnet_b2", hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # Remove the classifier head
        self.backbone.classifier = nn.Identity()
        
        # Get the feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze early layers for memory efficiency
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in ['blocks.0', 'blocks.1', 'blocks.2']):
                param.requires_grad = False
        
        # Project to desired dimension
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Extract features using EfficientNet
        features = self.backbone(images)  # [batch, feature_dim]
        
        # Project to desired dimension
        image_features = self.projection(features)  # [batch, hidden_dim]
        return image_features

class MultimodalPricePredictionModel(nn.Module):
    """Sophisticated multimodal model for price prediction"""
    
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        image_model_name: str = "efficientnet_b2",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Text and Image encoders
        self.text_encoder = TextEncoder(text_model_name, hidden_dim, dropout)
        self.image_encoder = ImageEncoder(image_model_name, hidden_dim, dropout)
        
        # Cross-attention mechanism
        self.cross_attention = MultiHeadCrossAttention(
            text_dim=hidden_dim,
            image_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Fusion and prediction layers
        self.fusion_layers = nn.Sequential(
            # Early fusion - concatenate text and image features
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
        )
        
        # Price prediction head with multiple outputs for ensemble-like behavior
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout // 4),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in [self.fusion_layers, self.price_predictor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        # Encode text and images
        text_features = self.text_encoder(input_ids, attention_mask)  # [batch, hidden_dim]
        image_features = self.image_encoder(images)  # [batch, hidden_dim]
        
        # Apply cross-attention (text attending to image)
        attended_text = self.cross_attention(text_features, image_features)
        
        # Concatenate attended text with original image features
        fused_features = torch.cat([attended_text, image_features], dim=1)  # [batch, hidden_dim*2]
        
        # Apply fusion layers
        fused_output = self.fusion_layers(fused_features)  # [batch, hidden_dim//2]
        
        # Predict price
        price_pred = self.price_predictor(fused_output)  # [batch, 1]
        
        # Ensure positive prices using softplus activation
        price_pred = F.softplus(price_pred) + 1e-6
        
        return price_pred.squeeze(-1)  # [batch]

class ModelConfig:
    """Configuration class for the multimodal model"""
    
    def __init__(self):
        # Model architecture
        self.text_model_name = "distilbert-base-uncased"
        self.image_model_name = "efficientnet_b0"
        self.hidden_dim = 512
        self.num_attention_heads = 8
        self.dropout = 0.3
        
        # Training parameters optimized for RTX 3050
        self.batch_size = 16  # Reduced for 6GB VRAM
        self.gradient_accumulation_steps = 4  # Effective batch size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.num_epochs = 10
        self.warmup_steps = 1000
        
        # Data parameters
        self.max_text_length = 256  # Reduced to save memory
        self.image_size = 224
        self.num_workers = 2  # Conservative for stability
        
        # Mixed precision training
        self.use_amp = True
        
        # Model saving
        self.save_steps = 2000
        self.eval_steps = 1000
        
        # Early stopping
        self.patience = 5
        self.min_delta = 0.001

def create_model(config: ModelConfig) -> MultimodalPricePredictionModel:
    """Factory function to create the multimodal model"""
    model = MultimodalPricePredictionModel(
        text_model_name=config.text_model_name,
        image_model_name=config.image_model_name,
        hidden_dim=config.hidden_dim,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout
    )
    return model

if __name__ == "__main__":
    # Test model creation and forward pass
    config = ModelConfig()
    model = create_model(config)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, config.max_text_length)).to(device)
    attention_mask = torch.ones(batch_size, config.max_text_length).to(device)
    images = torch.randn(batch_size, 3, config.image_size, config.image_size).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, images)
        print(f"Output shape: {outputs.shape}")
        print(f"Sample predictions: {outputs[:5].cpu().numpy()}")