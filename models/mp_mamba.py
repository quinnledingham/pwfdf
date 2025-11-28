import torch
import torch.nn as nn
from mamba_ssm import Mamba 

import torch
import torch.nn as nn
# Assuming Mamba is correctly imported, e.g., from mamba_ssm import Mamba

from models.mamba import threat_score_loss

import torch
import torch.nn as nn
# Assuming Mamba is correctly imported

class InsaneMambaModel(nn.Module):
    def __init__(self, features, duration, d_model=64, dropout=0.1):
        super().__init__()
        self.name = 'InsaneMambaModel'
        self.d_model = d_model
        
        # ... (Feature index finding remains the same) ...
        self.T_idx = features.index('PropHM23')
        self.F_idx = features.index('dNBR/1000')
        self.S_idx = features.index('KF')
        duration_map = {'15min': 'Acc015_mm', '30min': 'Acc030_mm', '60min': 'Acc060_mm'}
        feature_name = duration_map[duration]
        self.R_idx = features.index(feature_name) if feature_name in features else 0
        self.feature_indices = [self.T_idx, self.F_idx, self.S_idx, self.R_idx]
        
        # Feature Projection/Embedding (1D feature -> d_model)
        self.proj_T = nn.Linear(1, d_model)
        self.proj_F = nn.Linear(1, d_model)
        self.proj_S = nn.Linear(1, d_model)
        self.proj_R = nn.Linear(1, d_model)
        
        # Mamba Blocks: d_model=64
        self.m1 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 
        self.m2 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 
        self.m3 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 
        self.m4 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 

        # Global Mamba: d_model=256
        self.global_mamba = Mamba(d_model=4 * d_model, d_state=16, d_conv=4, expand=2,)
        
        self.dropout = nn.Dropout(dropout) 
        
        # Output Head: Input shape will be (B, 4*d_model) after S=1 pooling
        self.output_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1) 
        )

    def forward(self, x, target=None):
        # x is (Batch, Features)
        
        # 1. Slicing and Sequence Dimension Injection (Input: (B, F) -> Output: (B, 1, 1))
        # Select feature and add sequence dimension (dim=1) and feature dimension (dim=2)
        
        # x[:, self.T_idx] -> (B,)
        # x[:, self.T_idx].unsqueeze(-1) -> (B, 1)
        # x[:, self.T_idx].unsqueeze(-1).unsqueeze(-1) -> (B, 1, 1) - Correct shape for Mamba input
        
        x_T_seq = x[:, self.T_idx].unsqueeze(-1).unsqueeze(-1)
        x_F_seq = x[:, self.F_idx].unsqueeze(-1).unsqueeze(-1)
        x_S_seq = x[:, self.S_idx].unsqueeze(-1).unsqueeze(-1)
        x_R_seq = x[:, self.R_idx].unsqueeze(-1).unsqueeze(-1)

        # 2. Projection (Input: (B, 1, 1) -> Output: (B, 1, d_model))
        x_T_proj = self.dropout(self.proj_T(x_T_seq)) 
        x_F_proj = self.dropout(self.proj_F(x_F_seq))
        x_S_proj = self.dropout(self.proj_S(x_S_seq))
        x_R_proj = self.dropout(self.proj_R(x_R_seq))
        
        # 3. Apply Mamba Blocks (Output: (B, 1, d_model))
        x_T = self.dropout(self.m1(x_T_proj))
        x_F = self.dropout(self.m2(x_F_proj))
        x_S = self.dropout(self.m3(x_S_proj))
        x_R = self.dropout(self.m4(x_R_proj))
        
        # 4. Concatenate and apply Global Mamba
        # Output: (B, 1, 4 * d_model)
        x_combined = torch.cat([x_T, x_F, x_S, x_R], dim=-1) 
        x_combined = self.dropout(x_combined)
        
        x_out = self.global_mamba(x_combined) # Output: (B, 1, 4 * d_model)
        
        # 5. Pooling and Output Head
        # Temporal Pooling: Since S=1, simply squeeze the sequence dimension (dim=1)
        x_pooled = x_out.squeeze(dim=1) # Output: (B, 4 * d_model)
        
        # Output Head for final prediction
        output = self.output_head(x_pooled)
        
        if target != None:
            loss = threat_score_loss(output, target)
        else:
            loss = None

        return torch.sigmoid(output), loss

class MultiPathwayHybridModel_og(nn.Module):
    def __init__(self, features, pos_weight, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'MultiPathwayHybrid'
        self.spatial = False
        
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
                    nn.Linear(input_dim * 2, 4) # Output feature size is 8
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
            'Fire': 4, 'Terrain': 4, 'Burn': 4, 'Soil': 4, 
            'Rain_Accumulation': d_model, 'Rain_Intensity': d_model, 'Storm': 4
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
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def _mamba_forward(self, x, pathway_modules):
        """Dedicated forward pass for Mamba pathways."""
        
        # x shape: (batch_size, feature_dim)
        x = x.unsqueeze(1) # (batch_size, 1, feature_dim) - Mamba expects a sequence dimension
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

    def forward(self, x, target=None):
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
            
        combined = torch.cat(batch_outputs, dim=1)
        output = self.combined_head(combined).squeeze(-1)

        if target != None:
            loss = threat_score_loss(output, target)
        else:
            loss = None

        return torch.sigmoid(output), loss

class MultiPathwayHybridModel(nn.Module):
    def __init__(self, features, input_dim=16, d_model=64, n_layers=4, dropout=0.1, pathway_output_dim=8):
        super().__init__()
        
        # --- Model Configuration ---
        self.input_dim = input_dim
        self.d_model = d_model
        self.pathway_output_dim = pathway_output_dim  # <--- NEW HYPERPARAMETER
        self.duration = '15min'
        self.name = 'MultiPathwayHybrid_LearnedImportance'
        self.all_features = features

        # --- 1. Define and Clean Feature Groups 🧼 (Unchanged Logic) ---
        initial_feature_groups = {
            'Fire': ['Fire_ID', 'Fire_SegID'],
            'Terrain': ['PropHM23', 'ContributingArea_km2'],
            'Burn': ['dNBR/1000', 'PropHM23'],
            'Soil': ['KF'],
            'Rain_Accumulation': ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm'],
            'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            'Storm': ['StormDur_H', 'GaugeDist_m']
        }
        
        self.feature_groups = {}
        for group_name, group_list in initial_feature_groups.items():
            filtered_list = [feat for feat in group_list if feat in self.all_features]
            if filtered_list:
                self.feature_groups[group_name] = filtered_list

        # --- 2. Determine Feature Indices and Pathway Dimensions ---
        self.group_indices = {}
        self.pathway_modules = nn.ModuleDict()
        
        for group_name, group_list in self.feature_groups.items():
            indices = [self.all_features.index(feat) for feat in group_list]
            self.group_indices[group_name] = indices
            group_input_dim = len(indices)
            
            # Define specialized pathways for each group
            if group_name in ['Fire', 'Terrain', 'Soil', 'Storm']:
                # Simple MLP/Linear layer for static/simple features
                # Output layer now projects to pathway_output_dim
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.Linear(group_input_dim, group_input_dim * 2),
                    nn.ReLU(),
                    nn.Linear(group_input_dim * 2, self.pathway_output_dim) # <--- MODIFIED
                )
            
            elif group_name in ['Burn']:
                # Dedicated small pathway
                # Output layer now projects to pathway_output_dim
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.Linear(group_input_dim, self.pathway_output_dim) # <--- MODIFIED
                )
            
            elif group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                # Mamba pathway for sequence-like/complex time-series features
                mamba_input_proj = nn.Linear(group_input_dim, d_model)
                mamba_layers = nn.ModuleList([
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 
                    for _ in range(n_layers)
                ])
                mamba_norms = nn.ModuleList([
                    nn.LayerNorm(d_model)
                    for _ in range(n_layers)
                ])
                # --- NEW: Final projection layer for Mamba output ---
                mamba_final_proj = nn.Linear(d_model, self.pathway_output_dim) 
                
                self.pathway_modules[group_name] = nn.ModuleDict({
                    'proj': mamba_input_proj,
                    'layers': mamba_layers,
                    'norms': mamba_norms,
                    'final_proj': mamba_final_proj # <--- NEW
                })

        # --- 3. Combined Output Head --- 
        
        # All active groups now output the same dimension (pathway_output_dim)
        num_active_groups = len(self.feature_groups)
        total_combined_dim = num_active_groups * self.pathway_output_dim
        
        #print(f"Total Active Groups: {num_active_groups}")
        #print(f"Pathway Output Dim: {self.pathway_output_dim}")
        #print(f"Total Combined Dim for Head: {total_combined_dim}")
        
        self.combined_head = nn.Sequential(
            # The first linear layer will learn the importance of each group's output vector
            nn.Linear(total_combined_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)

    # --- Modified Mamba Forward Pass ---
    def _mamba_forward(self, x, pathway_modules):
        """Dedicated forward pass for Mamba pathways."""
        
        # x shape: (batch_size, feature_dim)
        x = x.unsqueeze(1) # (batch_size, 1, feature_dim)
        
        # Input projection
        x = pathway_modules['proj'](x) # (batch_size, 1, d_model)
        
        # Mamba layers
        for mamba_layer, norm in zip(pathway_modules['layers'], pathway_modules['norms']):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x
            
        # Global pooling over sequence dimension: (batch_size, d_model)
        x = x.mean(dim=1) 
        
        # Final projection to the uniform output dimension: (batch_size, pathway_output_dim)
        x = pathway_modules['final_proj'](x) # <--- MODIFIED
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