# Siamese-Transformer Network for Signature Verification

## 📋 Project Overview

This project implements a state-of-the-art signature verification system using a hybrid Siamese-Transformer architecture. The system is designed to authenticate handwritten signatures by learning similarity patterns between genuine signatures and detecting forgeries with high accuracy.

### Key Features

- **Hybrid CNN-Transformer Architecture**: Combines local feature extraction with global context modeling
- **Siamese Network Design**: Learns similarity metrics rather than classification
- **Triplet Loss Training**: Optimizes embedding space for better signature discrimination
- **Advanced Preprocessing**: Handles signature-specific challenges like noise and contrast variations

---

## 🏗️ Model Architecture

### High-Level Architecture Diagram

```
Input Signature Images (224×224×1)
          │
          ▼
┌─────────────────────────────────────┐
│        HybridBackbone               │
│  ┌─────────────────────────────────┐│
│  │     CNN Front-end               ││
│  │  Conv2d(1→32) + BN + ReLU      ││
│  │  Conv2d(32→64) + BN + ReLU     ││
│  │  MaxPool2d(3×3, stride=2)      ││
│  │  Conv2d(64→128) + BN + ReLU    ││
│  │  Conv2d(128→512, 16×16)        ││
│  └─────────────────────────────────┘│
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────────┐│
│  │   Flatten & Add CLS Token      ││
│  │   Shape: (B, 197, 512)         ││
│  │   + Positional Embeddings      ││
│  └─────────────────────────────────┘│
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────────┐│
│  │    Transformer Encoder          ││
│  │    8 Layers × 8 Attention Heads││
│  │    Hidden Size: 512             ││
│  │    MLP Dimension: 2048          ││
│  └─────────────────────────────────┘│
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────────┐│
│  │   Stroke-Aware Attention        ││
│  │   + Feature Refinement          ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
          │
          ▼
    512-D Embedding Vector

┌─────────────────────────────────────┐
│      Siamese Architecture           │
│                                     │
│  Signature A ──► Backbone ──► Emb_A │
│                    │                │
│  Signature B ──► Shared ────► Emb_B │
│                                     │
│  Distance(Emb_A, Emb_B) ──► Score   │
└─────────────────────────────────────┘
```

### Architecture Components

#### 1. **CNN Front-end (Local Feature Extraction)**

```python
# Progressive feature extraction
Conv2d(1 → 32, 7×7, stride=2) + BatchNorm + ReLU
Conv2d(32 → 64, 3×3) + BatchNorm + ReLU + MaxPool
Conv2d(64 → 128, 3×3) + BatchNorm + ReLU
Conv2d(128 → 512, 16×16, stride=16)  # Patch embedding
```

**Reasoning**:

- **7×7 initial conv**: Captures stroke-level features (pen width, direction)
- **Progressive channels**: 32→64→128→512 allows hierarchical feature learning
- **Final 16×16 conv**: Converts feature maps to patch embeddings for transformer

#### 2. **Transformer Back-end (Global Context)**

```python
# Configuration
Hidden Size: 512
Layers: 8
Attention Heads: 8
MLP Dimension: 2048
Dropout: 0.1
```

**Reasoning**:

- **8 layers**: Sufficient depth for signature complexity without overfitting
- **512 embedding**: Balances model capacity with computational efficiency
- **8 attention heads**: Multiple attention patterns for different signature aspects
- **CLS token**: Global signature representation for similarity comparison

#### 3. **Stroke-Aware Attention Layer**

```python
# Additional attention mechanism for signature-specific patterns
MultiheadAttention(embed_dim=512, num_heads=4)
+ Feature refinement network
```

**Reasoning**: Signatures have unique stroke patterns that benefit from specialized attention mechanisms.

---

## 💾 Data Pipeline

### Dataset Structure

```
data/
├── train/
│   ├── user_001/
│   │   ├── genuine_01.png
│   │   ├── genuine_02.png
│   │   ├── forged_01.png
│   │   └── forged_02.png
│   └── user_002/
│       └── ...
└── val/
    └── [same structure]
```

### Triplet Generation Strategy

For each training sample, we generate triplets:

- **Anchor**: Genuine signature from user X
- **Positive**: Different genuine signature from user X
- **Negative**: 70% forgery from user X, 30% genuine from user Y

**Reasoning**:

- **70% forgery preference**: Forgeries are harder negatives, improving discrimination
- **30% inter-user**: Maintains general signature vs non-signature distinction

### Preprocessing Pipeline

```python
1. Grayscale conversion
2. Contrast enhancement (α=1.2, β=10)
3. Noise reduction (median blur)
4. Adaptive thresholding for binarization
5. Data augmentation (training only):
   - Random rotation (-10° to +10°)
   - Slight scaling (0.95 to 1.05)
6. Resize to 224×224
7. Normalization (0-1 range)
```

**Reasoning**:

- **Contrast enhancement**: Improves stroke visibility
- **Adaptive thresholding**: Better than global thresholding for varying lighting
- **Conservative augmentation**: Preserves signature integrity while adding robustness

---

## 🎯 Training Strategy

### Training Configuration

```python
MODEL_CONFIG = {
    'embed_dim': 512,      # Increased from 256
    'depth': 8,            # Increased from 4
    'heads': 8,            # Increased from 4
    'mlp_dim': 2048        # Increased from 1024
}

TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,      # Reduced for stability
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'margin': 0.5,         # Triplet loss margin
    'warmup_epochs': 10,
    'patience': 15
}
```

### Training Process

#### 1. **Triplet Loss Function**

```python
TripletMarginLoss(margin=0.5, p=2)  # Euclidean distance
```

**Reasoning**: Encourages anchor-positive distance < anchor-negative distance by margin of 0.5

#### 2. **Learning Rate Schedule**

- **Warmup**: Linear increase over 10 epochs
- **Main training**: Cosine annealing decay
- **AdamW optimizer**: Better regularization than Adam

#### 3. **Training Enhancements**

- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- **Early stopping**: Prevents overfitting (patience=15)
- **Mixed precision**: Faster training with maintained accuracy

### Key Metrics Tracked

- **Triplet Accuracy**: Percentage of correctly ordered triplets
- **Training/Validation Loss**: Triplet margin loss
- **Learning Rate**: For monitoring schedule

---

## 🔧 Implementation Details

### File Structure

```
project/
├── model.py           # Architecture definitions
├── utils.py           # Preprocessing functions
├── data_loader.py     # Dataset and triplet generation
├── train.py           # Training script
└── trained_models/    # Model checkpoints
```

### Key Classes

#### `HybridBackbone`

- Core feature extraction network
- CNN → Transformer pipeline
- Returns 512-dimensional embeddings

#### `SiameseTransformer`

- Main model class
- Wraps HybridBackbone for dual input processing
- Handles forward pass for signature pairs

#### `TripletSignatureDataset`

- Custom dataset class
- Dynamic triplet generation
- Handles genuine/forged signature sampling

---

## 📊 Technical Decisions & Reasoning

### Why Siamese Architecture?

1. **Similarity Learning**: More suitable than classification for verification tasks
2. **Few-shot Capability**: Works with limited samples per user
3. **Metric Learning**: Learns meaningful distance metrics in embedding space

### Why CNN-Transformer Hybrid?

1. **Local Features**: CNN captures stroke-level details (thickness, direction)
2. **Global Context**: Transformer models signature flow and overall structure
3. **Best of Both**: Combines spatial inductive bias with attention mechanisms

### Why Triplet Loss?

1. **Relative Comparisons**: More stable than contrastive loss
2. **Hard Negative Mining**: Naturally focuses on difficult cases
3. **Embedding Quality**: Produces well-separated clusters in embedding space

---

## 🚀 Current Status

### ✅ Completed

- [x] Hybrid CNN-Transformer architecture implementation
- [x] Siamese network wrapper
- [x] Advanced preprocessing pipeline
- [x] Triplet dataset generation
- [x] Training script with metrics tracking
- [x] Learning rate scheduling and optimization
- [x] Model checkpointing and early stopping

### 🔄 In Progress

- [ ] Model architecture debugging (position embedding fix)
- [ ] Hyperparameter validation
- [ ] Initial training runs

### ⏸️ Known Issues

1. **Position Embedding Bug**: Dimensional mismatch between CNN output patches and position embeddings
2. **Memory Optimization**: Large model may require gradient checkpointing
3. **Batch Size Tuning**: May need adjustment based on GPU memory

---

## 📈 Upcoming Milestones

### Phase 1: Core Implementation (Current)

- [ ] Fix position embedding calculation
- [ ] Validate model forward pass
- [ ] Run initial training experiments
- [ ] Establish baseline performance metrics

### Phase 2: Training & Optimization (Next 2 weeks)

- [ ] Train on signature dataset
- [ ] Hyperparameter tuning (learning rate, batch size, margin)
- [ ] Implement advanced evaluation metrics (EER, FAR, FRR)
- [ ] Model performance analysis and debugging

### Phase 3: Advanced Features (Future)

- [ ] Few-shot learning capabilities (based on research paper)
- [ ] Meta-learning training paradigm
- [ ] Advanced loss functions (Focal Loss, Center Loss)
- [ ] Model ensemble techniques
- [ ] Online hard negative mining

### Phase 4: Production Readiness

- [ ] Model optimization and quantization
- [ ] Inference pipeline development
- [ ] API wrapper creation
- [ ] Performance benchmarking
- [ ] Documentation and deployment guide

---

## 🎯 Success Metrics

### Technical Metrics

- **Equal Error Rate (EER)**: < 5% (industry standard)
- **False Acceptance Rate (FAR)**: < 1% at operating point
- **False Rejection Rate (FRR)**: < 5% at operating point
- **Training Convergence**: Stable loss decrease over epochs

### Performance Metrics

- **Inference Time**: < 100ms per signature pair
- **Model Size**: < 50MB for deployment
- **GPU Memory**: < 4GB during training
- **Accuracy**: > 95% on test set

---

## 📚 Research Alignment

Based on recent paper "Siamese-Transformer Network for Offline Handwritten Signature Verification using Few-shot":

### ✅ Aligned Aspects

- Hybrid CNN-Transformer architecture
- Siamese network foundation
- Focus on contextual relationships
- Limited data handling capability

### 🔄 Future Enhancements

- Few-shot learning integration
- Meta-learning training paradigm
- Episode-based training structure
- Prototypical network components

---

## 🤝 Meeting Discussion Points

### Technical Architecture

1. **Model capacity**: 512-dim embeddings vs alternatives
2. **Training stability**: Gradient clipping and warmup strategy
3. **Memory requirements**: Batch size and model size trade-offs

### Implementation Decisions

1. **Triplet vs Few-shot**: Current approach vs research paper alignment
2. **Data augmentation**: Signature-specific vs general techniques
3. **Loss function**: Triplet margin vs advanced alternatives

### Next Steps Priority

1. **Immediate fixes**: Position embedding bug resolution
2. **Training pipeline**: Dataset preparation and initial experiments
3. **Future roadmap**: Few-shot learning integration timeline

---

## 📞 Contact & Resources

- **Project Repository**: [Internal Git Repository]
- **Model Checkpoints**: `./trained_models/`
- **Training Logs**: Available via TensorBoard/Weights & Biases
- **Research Papers**: Shared documentation folder

---

_Last Updated: June 24, 2025_  
_Status: Implementation Phase - Ready for Initial Training_
