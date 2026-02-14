# =============================================================================
# ENHANCED TEMPORAL CONFORMER WITH BiLSTM AND SELF-SUPERVISED PRETRAINING
# With Weighted L1 + Coordinate Normalize Loss for BOTH SSL and Supervised Training
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("=" * 80)
print("ENHANCED TEMPORAL CONFORMER WITH BiLSTM AND WEIGHTED L1 + COORDINATE LOSS")
print("SWELLEX96 EXPERIMENT - FULL U-SHAPE TRACK ANALYSIS")
print("=" * 80)

# =============================================================================
# COMBINED LOSS FUNCTION: WEIGHTED L1 + COORDINATE NORMALIZE LOSS
# =============================================================================

class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that emphasizes different error regions
    Higher weight for larger errors to prevent underfitting on difficult samples
    """
    def __init__(self, reduction='mean'):
        super(WeightedL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        predictions: tensor of shape [batch_size, 1] or [batch_size]
        targets: tensor of shape [batch_size]
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        absolute_errors = torch.abs(predictions - targets)

        # Dynamic weighting based on error magnitude
        # Higher weight for larger errors to focus on difficult samples
        weights = 1.0 + torch.tanh(absolute_errors / 100.0)  # Scale for range in meters

        weighted_errors = weights * absolute_errors

        if self.reduction == 'mean':
            return weighted_errors.mean()
        elif self.reduction == 'sum':
            return weighted_errors.sum()
        else:
            return weighted_errors

class CoordinateNormalizeLoss(nn.Module):
    """
    Coordinate-aware normalization loss that considers geometric relationships
    and normalizes errors based on the coordinate space
    """
    def __init__(self, reduction='mean'):
        super(CoordinateNormalizeLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        predictions: tensor of shape [batch_size, 1] or [batch_size]
        targets: tensor of shape [batch_size]
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # Normalized error considering the coordinate scale
        errors = predictions - targets

        # Coordinate-aware normalization
        # Normalize by target magnitude to get relative error
        normalized_errors = errors / (targets + 1e-8)  # Add epsilon to avoid division by zero

        # Use hyperbolic tangent to bound the normalized errors
        coordinate_loss = torch.mean(torch.tanh(torch.abs(normalized_errors)) * torch.abs(errors))

        return coordinate_loss

class CombinedWeightedL1CoordinateLoss(nn.Module):
    """
    Combined loss: Weighted L1 + Coordinate Normalize Loss
    This provides robust error handling with geometric awareness
    """
    def __init__(self, l1_weight=0.7, coordinate_weight=0.3, reduction='mean'):
        super(CombinedWeightedL1CoordinateLoss, self).__init__()
        self.l1_weight = l1_weight
        self.coordinate_weight = coordinate_weight
        self.weighted_l1_loss = WeightedL1Loss(reduction)
        self.coordinate_loss = CoordinateNormalizeLoss(reduction)

    def forward(self, predictions, targets):
        """
        predictions: tensor of shape [batch_size, 1]
        targets: tensor of shape [batch_size]
        """
        # Ensure predictions are 1D for loss computation
        predictions_1d = predictions.squeeze()

        # Weighted L1 loss
        l1_loss = self.weighted_l1_loss(predictions_1d, targets)

        # Coordinate normalize loss
        coord_loss = self.coordinate_loss(predictions_1d, targets)

        # Combined loss
        total_loss = (self.l1_weight * l1_loss +
                     self.coordinate_weight * coord_loss)

        return total_loss, l1_loss, coord_loss

# =============================================================================
# SELF-SUPERVISED LEARNING COMPONENTS - IMPROVED MASKED MODELING
# =============================================================================

class MaskedModelingPretrainer:
    """Proper self-supervised pretraining using masked sequence modeling with reconstruction"""
    def __init__(self, model, mask_ratio=0.20, input_features=148, device='cuda'):
        self.model = model
        self.mask_ratio = mask_ratio
        self.input_features = input_features
        self.device = device

        # Use the same CombinedWeightedL1CoordinateLoss for SSL
        self.reconstruction_loss = CombinedWeightedL1CoordinateLoss(l1_weight=0.7, coordinate_weight=0.3)

        # Add reconstruction head for masked modeling - reconstruct single time step
        self.reconstruction_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, input_features)  # Reconstruct single time step features
        ).to(device)

    def random_masking(self, x):
        """Apply random masking to sequences with proper reconstruction targets"""
        batch_size, seq_len, features = x.shape

        # Create random mask - mask entire time steps
        num_masked = int(seq_len * self.mask_ratio)
        mask_indices = torch.randperm(seq_len)[:num_masked].to(self.device)

        # Create masked input
        masked_x = x.clone()
        for idx in mask_indices:
            masked_x[:, idx, :] = 0  # Mask entire time step

        # Reconstruction target: masked positions only
        reconstruction_target = torch.zeros(batch_size, num_masked, features, device=self.device)
        for i, idx in enumerate(mask_indices):
            reconstruction_target[:, i, :] = x[:, idx, :]

        return masked_x, mask_indices, reconstruction_target

    def train_epoch(self, dataloader, optimizer, device):
        self.model.train()
        self.reconstruction_head.train()
        total_loss = 0
        total_l1_loss = 0
        total_coord_loss = 0
        batch_count = 0

        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_x = batch_data[0]
            else:
                batch_x = batch_data

            batch_x = batch_x.to(self.device)

            # Apply masking - get masked input and reconstruction target
            masked_x, mask_indices, reconstruction_target = self.random_masking(batch_x)

            # Forward pass through encoder with masked input
            encoded = self.model.multimodal_proj(masked_x, self.model.feature_groups)
            encoded_pos = encoded + self.model.pos_encoding

            # BiLSTM processing
            lstm_out, _ = self.model.bilstm(encoded_pos)
            lstm_out = self.model.lstm_proj(lstm_out)
            lstm_out = self.model.lstm_norm(lstm_out)

            # Temporal processing with Conformer blocks
            temporal_out = lstm_out
            for block in self.model.blocks:
                temporal_out = block(temporal_out)

            # Get representations for masked positions
            batch_size = temporal_out.shape[0]
            num_masked = len(mask_indices)

            # Extract representations at masked positions
            masked_representations = torch.zeros(batch_size, num_masked, temporal_out.shape[-1], device=self.device)
            for i, idx in enumerate(mask_indices):
                masked_representations[:, i, :] = temporal_out[:, idx, :]

            # Flatten for reconstruction
            masked_flat = masked_representations.reshape(batch_size * num_masked, -1)

            # Reconstruction - predict the original features for masked positions
            reconstruction = self.reconstruction_head(masked_flat)

            # Reshape reconstruction to match target
            reconstruction = reconstruction.reshape(batch_size, num_masked, self.input_features)
            reconstruction_target_flat = reconstruction_target

            # Use CombinedWeightedL1CoordinateLoss for SSL reconstruction loss
            # We need to reshape for the loss function which expects 1D targets
            reconstruction_flat = reconstruction.reshape(batch_size * num_masked, self.input_features)
            target_flat = reconstruction_target_flat.reshape(batch_size * num_masked, self.input_features)

            # Calculate loss for each feature dimension and average
            feature_losses = []
            feature_l1_losses = []
            feature_coord_losses = []

            for feature_idx in range(self.input_features):
                feature_pred = reconstruction_flat[:, feature_idx].unsqueeze(1)
                feature_target = target_flat[:, feature_idx]

                feature_total_loss, feature_l1, feature_coord = self.reconstruction_loss(feature_pred, feature_target)
                feature_losses.append(feature_total_loss)
                feature_l1_losses.append(feature_l1)
                feature_coord_losses.append(feature_coord)

            # Average losses across all features
            loss = torch.mean(torch.stack(feature_losses))
            l1_loss_component = torch.mean(torch.stack(feature_l1_losses))
            coord_loss_component = torch.mean(torch.stack(feature_coord_losses))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.reconstruction_head.parameters()), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l1_loss += l1_loss_component.item()
            total_coord_loss += coord_loss_component.item()
            batch_count += 1

        return total_loss / batch_count, total_l1_loss / batch_count, total_coord_loss / batch_count

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

print("\n1. DATA LOADING AND PREPROCESSING...")

# Load datasets
df_unlabeled = pd.read_csv('/content/unlabeled_augmented_features_2400_samples.csv')
df_labeled = pd.read_csv('/content/new 3 sec without overlap label data.csv')

print(f"Unlabeled dataset shape: {df_unlabeled.shape}")
print(f"Labeled dataset shape: {df_labeled.shape}")

# Remove missing values
df_unlabeled = df_unlabeled.dropna()
df_labeled = df_labeled.dropna()

# Identify all common and unique features (excluding the label column for now)
all_labeled_cols_features = set(df_labeled.drop('range_label', axis=1, errors='ignore').columns)
all_unlabeled_cols_features = set(df_unlabeled.drop('range_label', axis=1, errors='ignore').columns)
common_feature_cols = sorted(list(all_labeled_cols_features.union(all_unlabeled_cols_features))) # Union of all features

# Ensure both dataframes have the same columns in the same order, filling missing with 0
for col in common_feature_cols:
    if col not in df_labeled.columns:
        df_labeled[col] = 0.0
    if col not in df_unlabeled.columns:
        df_unlabeled[col] = 0.0

# Reindex to ensure consistent column order across both dataframes
df_labeled = df_labeled[common_feature_cols + ['range_label']]
df_unlabeled = df_unlabeled[common_feature_cols + ['range_label']]

# Separate features and labels
X_unlabeled = df_unlabeled.drop('range_label', axis=1, errors='ignore')
X_labeled = df_labeled.drop('range_label', axis=1, errors='ignore')
true_ranges = df_labeled['range_label'].values

print(f"Labeled features shape: {X_labeled.shape}")
print(f"Unlabeled features shape: {X_unlabeled.shape}")
print(f"Range statistics - Min: {true_ranges.min():.2f}m, Max: {true_ranges.max():.2f}m, Mean: {true_ranges.mean():.2f}m")

# Convert to kilometers for better scaling
true_ranges_km = true_ranges / 1000.0
print(f"Range in km - Min: {true_ranges_km.min():.3f}km, Max: {true_ranges_km.max():.3f}km, Mean: {true_ranges_km.mean():.3f}km")

# =============================================================================
# 2. ROBUST FEATURE PROCESSING WITH TEMPORAL STRUCTURE
# =============================================================================

print("\n2. ROBUST FEATURE PROCESSING WITH TEMPORAL STRUCTURE...")

def separate_features_by_modality(X):
    """Separate features by modality while preserving physical meaning"""
    # Get available features in the dataset
    available_features = set(X.columns)

    scm_features = [col for col in X.columns if 'SCM_' in col]
    gcc_features = [col for col in X.columns if 'GCC_' in col]
    tdoa_features = [col for col in X.columns if 'TDOA_' in col]

    # Other features are those not in the above categories
    used_features = set(scm_features + gcc_features + tdoa_features)
    other_features = [col for col in X.columns if col not in used_features and col != 'range_label']

    # Separate SCM into real and imaginary parts
    scm_real = [col for col in scm_features if '_real' in col.lower()]
    scm_imag = [col for col in scm_features if '_imag' in col.lower()]

    # If no explicit real/imag markers, split by position
    if not scm_real and not scm_imag and scm_features:
        mid_point = len(scm_features) // 2
        scm_real = scm_features[:mid_point]
        scm_imag = scm_features[mid_point:]

    return {
        'scm_real': scm_real,
        'scm_imag': scm_imag,
        'gcc': gcc_features,
        'tdoa': tdoa_features,
        'other': other_features
    }

# Create unified feature groups for both datasets
feature_groups = separate_features_by_modality(pd.DataFrame(columns=common_feature_cols))

print(f"Unified feature groups: SCM-real: {len(feature_groups['scm_real'])}, "
      f"SCM-imag: {len(feature_groups['scm_imag'])}, "
      f"GCC: {len(feature_groups['gcc'])}, "
      f"TDOA: {len(feature_groups['tdoa'])}, "
      f"Other: {len(feature_groups['other'])}")

def robust_feature_scaling_per_modality(X, feature_groups):
    """Apply robust scaling per feature modality for better normalization"""
    X_scaled = X.copy()
    scalers = {}

    for modality, features in feature_groups.items():
        if features and len(features) > 0:
            # Check which features actually exist in the dataset
            existing_features = [f for f in features if f in X.columns]
            if existing_features:
                # Use RobustScaler for all features to handle outliers
                scaler = RobustScaler()
                X_scaled[existing_features] = scaler.fit_transform(X[existing_features])
                scalers[modality] = (scaler, existing_features)

    return X_scaled, scalers

# Apply robust scaling to both labeled and unlabeled data
X_labeled_scaled, labeled_scalers = robust_feature_scaling_per_modality(X_labeled, feature_groups)
X_unlabeled_scaled, unlabeled_scalers = robust_feature_scaling_per_modality(X_unlabeled, feature_groups)

# Scale target ranges using RobustScaler (better for outliers)
scaler_y = RobustScaler()
true_ranges_scaled = scaler_y.fit_transform(true_ranges_km.reshape(-1, 1)).flatten()

print("Robust feature scaling completed!")
print(f"Scaled target range: [{true_ranges_scaled.min():.3f}, {true_ranges_scaled.max():.3f}]")

# =============================================================================
# 3. TEMPORAL SEQUENCE PREPARATION WITH PROPER SPLITS
# =============================================================================

print("\n3. PREPARING TEMPORAL SEQUENCES WITH PROPER SPLITS...")

def create_temporal_sequences(features, targets=None, sequence_length=16):
    """
    Create sequences with preserved temporal and spatial structure
    Returns: sequences [batch, seq_len, features], targets [batch] (if provided)
    """
    num_samples = len(features)
    num_sequences = num_samples - sequence_length + 1

    sequences = np.zeros((num_sequences, sequence_length, features.shape[1]))
    sequence_targets = np.zeros(num_sequences) if targets is not None else None

    for i in range(num_sequences):
        # Preserve temporal structure: [seq_len, features]
        sequences[i] = features.iloc[i:i+sequence_length].values
        # Target is the range at the end of the sequence
        if targets is not None:
            sequence_targets[i] = targets[i+sequence_length-1]

    if targets is not None:
        return sequences, sequence_targets
    else:
        return sequences

# Create temporal sequences
sequence_length = 16

# Labeled sequences for supervised training
labeled_sequences, labeled_sequence_targets = create_temporal_sequences(
    X_labeled_scaled, true_ranges_scaled, sequence_length
)

# Unlabeled sequences for self-supervised pretraining
unlabeled_sequences = create_temporal_sequences(
    X_unlabeled_scaled, sequence_length=sequence_length
)

print(f"Created {len(labeled_sequences)} labeled temporal sequences")
print(f"Created {len(unlabeled_sequences)} unlabeled temporal sequences")
print(f"Sequence shape: {labeled_sequences.shape}")

# =============================================================================
# 4. PROPER DATA SPLITTING - 40% TRAIN, 20% VAL, 40% TEST
# =============================================================================

print("\n4. SPLITTING DATA INTO 40% TRAIN / 20% VAL / 40% TEST...")

# First split: separate test set (40%)
X_temp, X_test, y_temp, y_test = train_test_split(
    labeled_sequences, labeled_sequence_targets,
    test_size=0.40, random_state=42, shuffle=True
)

# Second split: separate train (40%) and validation (20%) from remaining 60%
# 20% of total is 33.33% of remaining 60% (20/60 ≈ 0.333)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.333, random_state=42, shuffle=True  # 20% of total data
)

print(f"Training samples: {len(X_train)} ({len(X_train)/len(labeled_sequences)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(labeled_sequences)*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(labeled_sequences)*100:.1f}%)")
print(f"Total labeled sequences: {len(labeled_sequences)}")

# =============================================================================
# 5. MODEL ARCHITECTURE WITH OPTIMIZED BiLSTM POSITION
# =============================================================================

print("\n5. BUILDING TEMPORAL CONFORMER WITH OPTIMIZED BiLSTM ARCHITECTURE...")

class FixedTemporalAttention(nn.Module):
    """Fixed multi-head attention for temporal sequences"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(FixedTemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x

class FixedTemporalConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(FixedTemporalConformerBlock, self).__init__()

        # Temporal attention
        self.attention = FixedTemporalAttention(d_model, num_heads, dropout)

        # Position-wise feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Temporal convolution
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Temporal attention
        x = self.attention(x)

        # Position-wise FFN with residual
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out

        # Temporal convolution with residual
        residual = x
        x_norm = self.conv_norm(x)
        x_conv = x_norm.transpose(1, 2)
        conv_out = self.conv(x_conv)
        conv_out = conv_out.transpose(1, 2)
        x = residual + conv_out

        return x

class FixedMultiModalProjection(nn.Module):
    """Project different modalities to common space"""
    def __init__(self, feature_groups, d_model=128):
        super(FixedMultiModalProjection, self).__init__()

        # Calculate input dimensions for each modality
        scm_dim = len(feature_groups['scm_real']) + len(feature_groups['scm_imag'])
        gcc_dim = len(feature_groups['gcc'])
        tdoa_dim = len(feature_groups['tdoa'])
        other_dim = len(feature_groups['other'])

        # Project each modality separately
        self.scm_proj = nn.Linear(scm_dim, d_model // 4)
        self.gcc_proj = nn.Linear(gcc_dim, d_model // 4)
        self.tdoa_proj = nn.Linear(tdoa_dim, d_model // 4)
        self.other_proj = nn.Linear(other_dim, d_model // 4)

        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_linear = nn.Linear(d_model, d_model)
        self.fusion_activation = nn.GELU()
        self.fusion_dropout = nn.Dropout(0.1)

    def forward(self, x, feature_groups):
        batch_size, seq_len, features = x.shape

        # Extract modalities using feature group indices
        start_idx = 0

        scm_real_len = len(feature_groups['scm_real'])
        scm_imag_len = len(feature_groups['scm_imag'])
        gcc_len = len(feature_groups['gcc'])
        tdoa_len = len(feature_groups['tdoa'])
        other_len = len(feature_groups['other'])

        scm_features = x[:, :, start_idx : start_idx + scm_real_len + scm_imag_len]
        start_idx += (scm_real_len + scm_imag_len)

        gcc_features = x[:, :, start_idx : start_idx + gcc_len]
        start_idx += gcc_len

        tdoa_features = x[:, :, start_idx : start_idx + tdoa_len]
        start_idx += tdoa_len

        other_features = x[:, :, start_idx : start_idx + other_len]

        # Project each modality
        scm_proj = self.scm_proj(scm_features)
        gcc_proj = self.gcc_proj(gcc_features)
        tdoa_proj = self.tdoa_proj(tdoa_features)
        other_proj = self.other_proj(other_features)

        # Concatenate and fuse
        fused = torch.cat([scm_proj, gcc_proj, tdoa_proj, other_proj], dim=-1)
        fused_norm = self.fusion_norm(fused)
        fused_out = self.fusion_linear(fused_norm)
        fused_out = self.fusion_activation(fused_out)
        fused_out = self.fusion_dropout(fused_out)

        return fused_out

class EnhancedTemporalConformerWithOptimizedBiLSTM(nn.Module):
    def __init__(self, input_features, sequence_length=16, d_model=128, num_heads=4,
                 num_blocks=3, d_ff=256, feature_groups=None, lstm_hidden_size=128, num_lstm_layers=2):
        super(EnhancedTemporalConformerWithOptimizedBiLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.feature_groups = feature_groups

        # Multi-modal projection
        self.multimodal_proj = FixedMultiModalProjection(feature_groups, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model))

        # BiLSTM layer BEFORE Conformer blocks for better temporal feature extraction
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # LSTM output projection to match d_model
        self.lstm_proj = nn.Linear(lstm_hidden_size * 2, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)

        # Temporal Conformer blocks after BiLSTM
        self.blocks = nn.ModuleList([
            FixedTemporalConformerBlock(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_blocks)
        ])

        # Global attention pooling instead of simple averaging
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.pool_norm = nn.LayerNorm(d_model)

        # Regression head with Weighted L1 + Coordinate Loss optimization
        self.regression_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)  # Single output for range prediction
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Multi-modal projection
        x_proj = self.multimodal_proj(x, self.feature_groups)

        # Add positional encoding
        x_pos = x_proj + self.pos_encoding

        # BiLSTM processing FIRST for enhanced temporal feature extraction
        lstm_out, _ = self.bilstm(x_pos)
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = self.lstm_norm(lstm_out)

        # Temporal processing with Conformer blocks AFTER BiLSTM
        temporal_out = lstm_out
        for block in self.blocks:
            temporal_out = block(temporal_out)

        # Attention-based pooling for better sequence aggregation
        query = temporal_out.mean(dim=1, keepdim=True)  # Global query
        attended_out, _ = self.attention_pool(query, temporal_out, temporal_out)
        pooled = attended_out.squeeze(1)

        # Final regression
        pooled_norm = self.regression_norm(pooled)
        prediction = self.regression_head(pooled_norm)

        return prediction

# =============================================================================
# 6. MODEL INITIALIZATION
# =============================================================================

print("\n6. INITIALIZING MODEL...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_features = X_labeled.shape[1]
model = EnhancedTemporalConformerWithOptimizedBiLSTM(
    input_features=input_features,
    sequence_length=sequence_length,
    d_model=128,
    num_heads=4,
    num_blocks=3,
    d_ff=256,
    feature_groups=feature_groups,
    lstm_hidden_size=128,
    num_lstm_layers=2
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
test_batch = torch.randn(2, sequence_length, input_features).to(device)
with torch.no_grad():
    test_output = model(test_batch)
print(f"Test forward pass successful! Output shape: {test_output.shape}")

# =============================================================================
# 7. SELF-SUPERVISED PRETRAINING WITH WEIGHTED L1 + COORDINATE LOSS
# =============================================================================

print("\n7. STARTING SELF-SUPERVISED PRETRAINING WITH WEIGHTED L1 + COORDINATE LOSS...")

# Create dataloaders
batch_size = 32

# Unlabeled data for pretraining
unlabeled_tensor = torch.FloatTensor(unlabeled_sequences)
unlabeled_dataset = TensorDataset(unlabeled_tensor)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

# Labeled data for supervised training
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Self-supervised pretraining with CombinedWeightedL1CoordinateLoss
pretrainer = MaskedModelingPretrainer(model, mask_ratio=0.20, input_features=input_features, device=device)
ssl_optimizer = optim.AdamW(list(model.parameters()) + list(pretrainer.reconstruction_head.parameters()),
                           lr=1e-4, weight_decay=1e-5)

ssl_epochs = 30  # Increased for better pretraining
ssl_losses = []
ssl_l1_losses = []
ssl_coord_losses = []

print("Starting self-supervised pretraining with Weighted L1 + Coordinate Loss...")
for epoch in range(ssl_epochs):
    ssl_loss, ssl_l1_loss, ssl_coord_loss = pretrainer.train_epoch(unlabeled_dataloader, ssl_optimizer, device)
    ssl_losses.append(ssl_loss)
    ssl_l1_losses.append(ssl_l1_loss)
    ssl_coord_losses.append(ssl_coord_loss)

    if (epoch + 1) % 5 == 0:
        print(f'SSL Epoch [{epoch+1}/{ssl_epochs}] - Total Loss: {ssl_loss:.4f}, L1: {ssl_l1_loss:.4f}, Coord: {ssl_coord_loss:.4f}')

print("Self-supervised pretraining completed!")

# =============================================================================
# 8. SUPERVISED FINE-TUNING WITH WEIGHTED L1 + COORDINATE LOSS
# =============================================================================

print("\n8. STARTING SUPERVISED FINE-TUNING WITH WEIGHTED L1 + COORDINATE LOSS...")

# Combined Weighted L1 + Coordinate Loss
criterion = CombinedWeightedL1CoordinateLoss(l1_weight=0.7, coordinate_weight=0.3)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

# Cosine annealing with warmup
def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr=2e-4):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

supervised_epochs = 120
warmup_epochs = 15
scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs, supervised_epochs)

# Training with early stopping
train_losses = []
val_losses = []
train_maes = []
val_maes = []
train_l1_losses = []
train_coord_losses = []
val_l1_losses = []
val_coord_losses = []
best_mae = float('inf')
patience = 20
patience_counter = 0

print("Starting supervised training with target MAE < 45m...")

for epoch in range(supervised_epochs):
    # Training
    model.train()
    epoch_train_loss = 0
    epoch_train_mae = 0
    epoch_train_l1 = 0
    epoch_train_coord = 0
    batch_count = 0

    for batch_X, batch_y in train_dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        predictions = model(batch_X)
        loss, l1_loss, coord_loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Calculate MAE
        predictions_1d = predictions.squeeze()
        mae = F.l1_loss(predictions_1d, batch_y).item()

        epoch_train_loss += loss.item()
        epoch_train_mae += mae
        epoch_train_l1 += l1_loss.item()
        epoch_train_coord += coord_loss.item()
        batch_count += 1

    # Validation
    model.eval()
    epoch_val_loss = 0
    epoch_val_mae = 0
    epoch_val_l1 = 0
    epoch_val_coord = 0
    val_batch_count = 0

    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            loss, l1_loss, coord_loss = criterion(predictions, batch_y)

            # Calculate MAE
            predictions_1d = predictions.squeeze()
            mae = F.l1_loss(predictions_1d, batch_y).item()

            epoch_val_loss += loss.item()
            epoch_val_mae += mae
            epoch_val_l1 += l1_loss.item()
            epoch_val_coord += coord_loss.item()
            val_batch_count += 1

    # Average metrics
    avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
    avg_train_mae = epoch_train_mae / batch_count if batch_count > 0 else 0
    avg_train_l1 = epoch_train_l1 / batch_count if batch_count > 0 else 0
    avg_train_coord = epoch_train_coord / batch_count if batch_count > 0 else 0

    avg_val_loss = epoch_val_loss / val_batch_count if val_batch_count > 0 else 0
    avg_val_mae = epoch_val_mae / val_batch_count if val_batch_count > 0 else 0
    avg_val_l1 = epoch_val_l1 / val_batch_count if val_batch_count > 0 else 0
    avg_val_coord = epoch_val_coord / val_batch_count if val_batch_count > 0 else 0

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_maes.append(avg_train_mae)
    val_maes.append(avg_val_mae)
    train_l1_losses.append(avg_train_l1)
    train_coord_losses.append(avg_train_coord)
    val_l1_losses.append(avg_val_l1)
    val_coord_losses.append(avg_val_coord)

    # Update scheduler
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # Early stopping check with MAE target
    if avg_val_mae < best_mae:
        best_mae = avg_val_mae
        torch.save(model.state_dict(), 'best_enhanced_model.pth')
        patience_counter = 0
        print(f"  ✓ New best model saved with MAE: {best_mae:.4f}")

        # Check if we've achieved target MAE
        if best_mae < 0.045:  # Scaled MAE target (45m in km scale)
            print(f"  🎯 TARGET ACHIEVED! MAE < 45m reached!")
    else:
        patience_counter += 1

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{supervised_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, Coord: {avg_train_coord:.4f})')
        print(f'  Train MAE: {avg_train_mae:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f} (L1: {avg_val_l1:.4f}, Coord: {avg_val_coord:.4f})')
        print(f'  Val MAE: {avg_val_mae:.4f}')
        print(f'  LR: {current_lr:.2e}, Best MAE: {best_mae:.4f}')
        print(f'  Patience: {patience_counter}/{patience}')

    # Early stopping
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Convert best MAE to meters for reporting
best_mae_meters = best_mae * 1000  # Convert from km scale to meters
print(f"Supervised fine-tuning completed! Best validation MAE: {best_mae_meters:.2f}m")

# Load best model
model.load_state_dict(torch.load('best_enhanced_model.pth'))

# =============================================================================
# 9. COMPREHENSIVE EVALUATION ON TEST SET
# =============================================================================

print("\n9. PERFORMING COMPREHENSIVE EVALUATION ON TEST SET...")

model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        batch_X = batch_X.to(device)
        predictions = model(batch_X)

        predictions_1d = predictions.squeeze().cpu().numpy()

        all_predictions.extend(predictions_1d)
        all_targets.extend(batch_y.numpy())

# Convert back to original scale (kilometers)
predictions_scaled = np.array(all_predictions).flatten()
targets_scaled = np.array(all_targets)

# Inverse transform to get kilometers
predictions_km = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
targets_km = scaler_y.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

# Convert to meters for final metrics
predictions_m = predictions_km * 1000
targets_m = targets_km * 1000

# Calculate metrics
mse = mean_squared_error(targets_m, predictions_m)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_m, predictions_m)
r2 = r2_score(targets_m, predictions_m)

print("\n" + "="*60)
print("FINAL TEST SET PERFORMANCE METRICS")
print("="*60)
print(f"MAE:  {mae:.2f} m")
print(f"RMSE: {rmse:.2f} m")
print(f"R²:   {r2:.4f}")
print(f"MSE:  {mse:.2f} m²")

# Check if target MAE is achieved
if mae < 45:
    print("🎯 TARGET ACHIEVED: MAE < 45m! 🎯")
else:
    print(f"⚠️  Target not reached: MAE = {mae:.2f}m (target: <45m)")

# =============================================================================
# 10. COMPREHENSIVE VISUALIZATION WITH FULL SWELLEX96 U-SHAPE TRACK
# =============================================================================

print("\n10. GENERATING COMPREHENSIVE SWELLEX96 U-SHAPE VISUALIZATIONS...")

# Create predictions for ALL labeled data to see the full U-shape trajectory
print("Generating predictions for full SWELLEX96 trajectory...")

# Prepare full labeled data for prediction
full_sequences, full_targets = create_temporal_sequences(
    X_labeled_scaled, true_ranges_scaled, sequence_length
)

full_sequences_tensor = torch.FloatTensor(full_sequences)
full_dataset = TensorDataset(full_sequences_tensor)
full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Generate predictions for full trajectory
model.eval()
full_predictions = []

with torch.no_grad():
    for batch_X in full_dataloader:
        batch_X = batch_X[0].to(device)
        predictions = model(batch_X)
        predictions_1d = predictions.squeeze().cpu().numpy()
        full_predictions.extend(predictions_1d)

# Convert full predictions back to original scale
full_predictions_scaled = np.array(full_predictions).flatten()
full_predictions_km = scaler_y.inverse_transform(full_predictions_scaled.reshape(-1, 1)).flatten()
full_predictions_m = full_predictions_km * 1000

# Get full true ranges (for the sequences)
full_true_ranges_m = true_ranges[sequence_length-1:]  # Align with sequence predictions

# Create proper time array for full SWELLEX96 trajectory (70 minutes total)
total_experiment_minutes = 70
full_time_minutes = np.linspace(0, total_experiment_minutes, len(full_predictions_m))

print(f"Full SWELLEX96 trajectory - Samples: {len(full_predictions_m)}, Time range: 0 to {total_experiment_minutes} minutes")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('SWELLEX96 - Enhanced Temporal Conformer + Optimized BiLSTM\nWeighted L1 + Coordinate Loss for BOTH SSL and Supervised - Full U-Shape Source Track Analysis',
             fontsize=16, fontweight='bold')

# 1. Training history
axes[0, 0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(val_losses, 'r-', label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Combined Loss')
axes[0, 0].set_title('Training History (Weighted L1 + Coordinate Loss)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes0_twin = axes[0, 0].twinx()
axes0_twin.plot(train_maes, 'g--', label='Train MAE', linewidth=1.5, alpha=0.7)
axes0_twin.plot(val_maes, 'm--', label='Val MAE', linewidth=1.5, alpha=0.7)
axes0_twin.set_ylabel('MAE')
axes0_twin.legend(loc='upper right')

# 2. Predictions vs True (meters) - Test set only
scatter = axes[0, 1].scatter(targets_m, predictions_m, alpha=0.6, c=targets_m,
                           cmap='viridis', s=30)
axes[0, 1].plot([targets_m.min(), targets_m.max()],
                [targets_m.min(), targets_m.max()], 'r--', linewidth=2)
axes[0, 1].set_xlabel('True Range (m)')
axes[0, 1].set_ylabel('Predicted Range (m)')
axes[0, 1].set_title(f'Test Set: Predictions vs True Values\nMAE: {mae:.2f}m, R²: {r2:.4f}')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='True Range (m)')

# 3. FULL SWELLEX96 U-SHAPE TRACK - This is the key plot!
axes[0, 2].plot(full_time_minutes, full_true_ranges_m, 'green', label='True Range',
                alpha=1.0, linewidth=4)
axes[0, 2].scatter(full_time_minutes, full_predictions_m, color='red',
                  label='Predicted Range', alpha=0.7, s=15, marker='o')

axes[0, 2].set_xlabel('Time (minutes)')
axes[0, 2].set_ylabel('Range (m)')
axes[0, 2].set_title('Range-Time Source Track: Ground Truth vs Predicted Labels')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlim(0, 70)

# 4. Residuals analysis (test set)
residuals_m = predictions_m - targets_m
axes[1, 0].scatter(predictions_m, residuals_m, alpha=0.6, c='blue', s=30)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Range (m)')
axes[1, 0].set_ylabel('Residuals (m)')
axes[1, 0].set_title('Test Set: Residuals vs Predicted Values')
axes[1, 0].grid(True, alpha=0.3)

# 5. Error distribution
axes[1, 1].hist(residuals_m, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error (m)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Test Set: Error Distribution\nMean Error: {residuals_m.mean():.2f}m ± {residuals_m.std():.2f}m')
axes[1, 1].grid(True, alpha=0.3)

# 6. SSL pretraining progress with both loss components
axes[1, 2].plot(ssl_losses, 'purple', linewidth=2, label='Total SSL Loss')
axes[1, 2].plot(ssl_l1_losses, 'orange', linewidth=1.5, linestyle='--', label='SSL L1 Loss', alpha=0.7)
axes[1, 2].plot(ssl_coord_losses, 'green', linewidth=1.5, linestyle='--', label='SSL Coord Loss', alpha=0.7)
axes[1, 2].set_xlabel('SSL Epoch')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].set_title('Self-Supervised Pretraining Progress\n(Weighted L1 + Coordinate Loss)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('swellex96_optimized_bilstm_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional summary statistics
error_percentage = (abs(residuals_m) / targets_m) * 100
print(f"\nSWELLEX96 PERFORMANCE ANALYSIS:")
print(f"Test Set Metrics:")
print(f"  MAE: {mae:.2f} m")
print(f"  RMSE: {rmse:.2f} m")
print(f"  R²: {r2:.4f}")
print(f"  Mean Absolute Percentage Error: {error_percentage.mean():.2f}%")
print(f"  Median Absolute Error: {np.median(abs(residuals_m)):.2f}m")
print(f"  Error Std: {residuals_m.std():.2f}m")
print(f"  95% Error Percentile: {np.percentile(abs(residuals_m), 95):.2f}m")

# Full trajectory statistics
full_residuals = full_predictions_m - full_true_ranges_m
full_mae = np.mean(np.abs(full_residuals))
full_rmse = np.sqrt(np.mean(full_residuals**2))

print(f"\nFull U-Shape Trajectory Metrics:")
print(f"  MAE: {full_mae:.2f} m")
print(f"  RMSE: {full_rmse:.2f} m")
print(f"  Samples: {len(full_predictions_m)}")
print(f"  Time range: 0 to {total_experiment_minutes} minutes")

print(f"\nModel Architecture Details:")
print(f"  BiLSTM Position: BEFORE Conformer blocks (optimized)")
print(f"  BiLSTM Layers: {model.bilstm.num_layers}")
print(f"  BiLSTM Hidden Size: {model.bilstm.hidden_size}")
print(f"  Conformer Blocks: {len(model.blocks)}")
print(f"  Loss Function: Weighted L1 + Coordinate Normalize Loss (BOTH SSL and Supervised)")
print(f"  SSL Method: Proper Masked Modeling with Weighted L1 + Coordinate Loss")

print(f"\nDataset Split Summary:")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(labeled_sequences)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(labeled_sequences)*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(labeled_sequences)*100:.1f}%)")
print(f"Unlabeled samples: {len(unlabeled_sequences)}")
print(f"Full trajectory samples: {len(full_predictions_m)}")

print("\n" + "="*80)
print("SWELLEX96 ENHANCED TEMPORAL CONFORMER + OPTIMIZED BiLSTM TRAINING COMPLETED!")
print(f"FINAL TEST MAE: {mae:.2f}m")
if mae < 45:
    print("🎯 TARGET MAE < 45m ACHIEVED SUCCESSFULLY! 🎯")
else:
    print(f"⚠️  Target MAE < 45m not reached (current: {mae:.2f}m)")
print("Full U-shape source track visualization generated!")
print("="*80)