import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MambaClassifier(nn.Module):
    def __init__(self, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'Mamba'

        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            #nn.Linear(d_model, d_model)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            #nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # For tabular data, we treat each sample as sequence of length 1
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Input projection
        x = self.input_proj(x)
        
        # Mamba layers
        for mamba_layer, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x  # Residual connection
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        return self.output_head(x).squeeze(-1)

class HybridMambaLogisticModel(nn.Module):
    def __init__(self, features, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'HybridMamba'
        
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = features
        
        # Get indices for rainfall vs non-rainfall features
        self.rainfall_indices = [self.all_features.index(feat) for feat in self.rainfall_features if feat in self.all_features]
        self.non_rainfall_indices = [i for i in range(len(self.all_features)) if i not in self.rainfall_indices]
        
        #print(f"Rainfall features ({len(self.rainfall_indices)}): {self.rainfall_features}")
        #print(f"Non-rainfall features ({len(self.non_rainfall_indices)}): {[self.all_features[i] for i in self.non_rainfall_indices]}")
        
        # Mamba pathway for non-rainfall features
        self.mamba_input_dim = len(self.non_rainfall_indices)
        self.mamba_input_proj = nn.Linear(self.mamba_input_dim, d_model)
        
        # Mamba backbone
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,)
            #nn.Linear(d_model, d_model)
            for _ in range(n_layers)
        ])
        
        # Layer normalization for Mamba
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # Logistic regression pathway for rainfall features
        self.logistic_input_dim = len(self.rainfall_indices)
        self.logistic_layer = nn.Sequential(
            nn.Linear(self.logistic_input_dim, 1),
            #nn.Sigmoid()
        )
        
        # Combined output head
        self.combined_head = nn.Sequential(
            nn.Linear(d_model + 1, 32),  # +1 for logistic output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            #nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def split_features(self, x):
        """Split input into rainfall and non-rainfall features"""
        # x shape: (batch_size, input_dim)
        non_rainfall_x = x[:, self.non_rainfall_indices]  # (batch_size, non_rainfall_dim)
        rainfall_x = x[:, self.rainfall_indices]          # (batch_size, rainfall_dim)
        return non_rainfall_x, rainfall_x
    
    def forward(self, x):
        non_rainfall_x, rainfall_x = self.split_features(x)
        
        mamba_out = self._mamba_forward(non_rainfall_x)
        logistic_out = self.logistic_layer(rainfall_x)  # (batch_size, 1)
        
        combined = torch.cat([mamba_out, logistic_out], dim=1)  # (batch_size, d_model + 1)
        
        # Final prediction
        output = self.combined_head(combined)
        return output.squeeze(-1)
    
    def _mamba_forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, mamba_input_dim)
        x = self.mamba_input_proj(x)  # (batch_size, 1, d_model)
        
        for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x  # Residual connection
        
        x = x.mean(dim=1)  # (batch_size, d_model)
        return x
    
class MultiPathwayHybridModel(nn.Module):
    def __init__(self, features, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'MultiPathwayHybrid'
        
        self.all_features = features
        
        # --- 1. Define Feature Groups ---
        self.feature_groups = {
            'Fire': ['Fire_ID', 'Fire_SegID'],
            'Terrain': ['PropHM23', 'ContributingArea_km2'],
            'Burn': ['dNBR/1000', 'PropHM23'],
            'Soil': ['KF'],
            'Rain_Accumulation': ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm'],
            'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            'Storm': ['StormDur_H', 'GaugeDist_m']
        }
        
        # --- 2. Determine Feature Indices and Pathway Dimensions ---
        self.group_indices = {}
        self.pathway_modules = nn.ModuleDict()
        
        for group_name, group_list in self.feature_groups.items():
            indices = [self.all_features.index(feat) for feat in group_list if feat in self.all_features]
            self.group_indices[group_name] = indices
            input_dim = len(indices)
            
            # Define specialized pathways for each group
            if group_name in ['Fire', 'Terrain', 'Soil', 'Storm']:
                # Simple MLP/Linear layer for static/simple features
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.Linear(input_dim, input_dim * 2),
                    nn.ReLU(),
                    nn.Linear(input_dim * 2, 8) # Output feature size is 8
                )
                
            elif group_name in ['Burn']:
                # Dedicated small pathway
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.Linear(input_dim, 4), # Output feature size is 4
                )
                
            elif group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                # Mamba pathway for sequence-like/complex time-series features
                # The Mamba layer will take the input dim and project to d_model, then process
                mamba_input_proj = nn.Linear(input_dim, d_model)
                mamba_layers = nn.ModuleList([
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 
                    for _ in range(n_layers)
                ])
                mamba_norms = nn.ModuleList([
                    nn.LayerNorm(d_model)
                    for _ in range(n_layers)
                ])
                
                # Store the components in a ModuleDict for easy access
                self.pathway_modules[group_name] = nn.ModuleDict({
                    'proj': mamba_input_proj,
                    'layers': mamba_layers,
                    'norms': mamba_norms
                })
        
        # Calculate the total concatenated dimension for the combined head
        # Fire (8) + Terrain (8) + Burn (4) + Soil (8) + Rain_Accumulation (d_model=64) + Rain_Intensity (d_model=64) + Storm (8)
        self.output_dim_map = {
            'Fire': 8, 'Terrain': 8, 'Burn': 4, 'Soil': 8, 
            'Rain_Accumulation': d_model, 'Rain_Intensity': d_model, 'Storm': 8
        }
        total_combined_dim = sum(self.output_dim_map.values())
        
        # --- 3. Combined Output Head ---
        self.combined_head = nn.Sequential(
            nn.Linear(total_combined_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            # nn.Sigmoid() for binary classification if needed
        )
        
        self.dropout = nn.Dropout(dropout)


    def _mamba_forward(self, x, pathway_modules):
        """Dedicated forward pass for Mamba pathways."""
        
        # x shape: (batch_size, feature_dim)
        x = x.unsqueeze(1) # (batch_size, 1, feature_dim) - Mamba expects a sequence dimension
        
        # Input projection
        x = pathway_modules['proj'](x) # (batch_size, 1, d_model)
        
        # Mamba layers
        for mamba_layer, norm in zip(pathway_modules['layers'], pathway_modules['norms']):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x # Residual connection
            
        x = x.mean(dim=1) # (batch_size, d_model) - Global pooling over sequence dimension
        return x


    def forward(self, x):
        batch_outputs = []
        
        for group_name, indices in self.group_indices.items():
            # 1. Split features
            group_x = x[:, indices]
            pathway_modules = self.pathway_modules[group_name]
            
            # 2. Process through the specialized pathway
            if group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                # Mamba Pathway
                out = self._mamba_forward(group_x, pathway_modules)
            else:
                # MLP/Linear Pathway
                out = pathway_modules(group_x)
            
            batch_outputs.append(out)
            
        # 3. Concatenate all pathway outputs
        combined = torch.cat(batch_outputs, dim=1)
        
        # 4. Final prediction
        output = self.combined_head(combined)
        return output.squeeze(-1)

# Dummy Mamba class for execution (replace with actual import)
class Mamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        # Placeholder Mamba logic: simple linear mapping
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.linear(x)

# Example Usage (requires PyTorch)
# features_list = ['Fire_ID', 'Fire_SegID', 'PropHM23', 'ContributingArea_km2', 'dNBR/1000', 'KF', 
#                  'Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm', 
#                  'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h', 
#                  'StormDur_H', 'GaugeDist_m']
# model = MultiPathwayHybridModel(features=features_list)
# dummy_input = torch.randn(32, len(features_list))
# output = model(dummy_input)
# print(output.shape) # Expected: torch.Size([32])