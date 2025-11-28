import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# -------- Local Branch Types -------- #

class MLPBranch(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


class AttentionBranch(nn.Module):
    def __init__(self, in_dim, embed_dim=32):
        super().__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x, _ = self.attn(x, x, x)
        return self.out(x.squeeze(1))


class MambaBranch(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.use_mamba = True
            self.embed = nn.Linear(in_dim, out_dim)
            self.mamba = Mamba(d_model=out_dim, d_state=16)
        except Exception:
            self.use_mamba = False
            self.mlp = MLPBranch(in_dim, out_dim=out_dim)

    def forward(self, x):
        if not self.use_mamba:
            return self.mlp(x)

        x = self.embed(x).unsqueeze(1)   # (B,1,d_model)
        x = self.mamba(x).squeeze(1)
        return x


# -------- Full Model With Global Mamba -------- #

class MultiBranchWithGlobalMamba(nn.Module):
    def __init__(
        self, 
        branch_out_dim=32,
        global_mamba_dim=32,
        output_dim=1
    ):
        super().__init__()

        # Feature groups (fixed order)
        self.feature_groups = {
            'Fire': ['Fire_ID', 'Fire_SegID'],
            'Terrain': ['PropHM23', 'ContributingArea_km2'],
            'Burn': ['dNBR/1000'],
            'Soil': ['KF'],
            'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            'Storm': ['StormDur_H', 'GaugeDist_m']
        }

        # Flatten the total feature order
        self.feature_order = []
        for g in self.feature_groups.values():
            self.feature_order.extend(g)

        branch_types = {
            'Fire': 'mlp',
            'Terrain': 'mlp',
            'Burn': 'mlp',
            'Soil': 'mlp',
            'Rain_Intensity': 'mamba',
            'Storm': 'mlp'
        }

        self.groups = list(self.feature_groups.keys())
        self.branches = nn.ModuleDict()
        self.name = 'MBGMamba'

        # Local branches
        for group in self.groups:
            in_dim = len(self.feature_groups[group])
            btype = branch_types[group].lower()

            if btype == "mlp":
                self.branches[group] = MLPBranch(in_dim, out_dim=branch_out_dim)
            elif btype == "attention":
                self.branches[group] = AttentionBranch(in_dim, embed_dim=branch_out_dim)
            elif btype == "mamba":
                self.branches[group] = MambaBranch(in_dim, out_dim=branch_out_dim)
            else:
                raise ValueError(f"Unknown branch type: {btype}")

        self.global_mamba = Mamba(d_model=global_mamba_dim, d_state=16)
        self.global_embed = nn.Linear(branch_out_dim, global_mamba_dim)

        # Final head
        self.head = nn.Sequential(
            nn.Linear(global_mamba_dim * len(self.groups), 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            #nn.Sigmoid(),
        )

    # -------------- NEW: Accepts x = [B, F] ---------------- #

    def forward(self, x):
        """
        x: tensor of shape (B, F) following self.feature_order
        """
        B = x.size(0)
        idx = 0
        emb_list = []

        # Slice the input tensor according to the feature groups
        for group in self.groups:
            group_feats = self.feature_groups[group]
            n = len(group_feats)

            x_slice = x[:, idx:idx+n]
            idx += n

            emb_list.append(self.branches[group](x_slice))

        # (B, num_groups, branch_out_dim)
        emb_seq = torch.stack(emb_list, dim=1)

        gx = self.global_embed(emb_seq)

        # If fallback attention, returns (B,G,D)
        gx = self.global_mamba(gx)

        gx = gx.reshape(B, -1)
        return self.head(gx).squeeze(-1)
