import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from tqdm import tqdm

from data import PWFDF_Data
from eval import evaluate, compare_params, evaluate_model, threat_score

from models.log_reg import Staley2017Model, LogisticRegression
from models.mamba import MambaClassifier, HybridMambaLogisticModel, HybridMambaLogisticModelPro, SpatialMambaHybridModel, SpatialMambaContextModel, HybridGroupedSpatialMambaModel, create_spatial_tensors, HybridMambaFeatureGroups, HybridGroupedSpatialMambaModel2
from models.new_mamba import MultiBranchWithGlobalMamba
from models.mp_mamba import MultiPathwayHybridModel, MultiPathwayHybridModel_og, InsaneMambaModel
from models.transformer import TransformerClassifier
from models.TSMixer import TSMixerClassifier, BestSimpleModel, Test, StaticMLPModel
from models.randomforest import RandomForestModel, train_random_forest

import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_file = './output/logs/results_seeds100.txt'

features = [
    'UTM_X', 'UTM_Y', 
    'Fire_ID', 'Fire_SegID',
    'GaugeDist_m', 
    'StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h', 
    'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
    'ContributingArea_km2', 
    'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 
    'Acc030_mm', 'Acc060_mm'
]

features_og = [
    'Fire_ID',
    'Fire_SegID',

    'PropHM23', 
    'ContributingArea_km2', 
   
    'dNBR/1000', 
    'KF',

    'Peak_I15_mm/h',
    'Peak_I30_mm/h',
    'Peak_I60_mm/h',
    'StormAvgI_mm/h',

    'StormDur_H',
    'GaugeDist_m', 
]

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    #format='%(asctime)s - %(levelname)s - %(message)s',
    format='%(message)s',
    handlers=[
        logging.FileHandler(output_file, encoding='utf-8'),
        #logging.StreamHandler()  # This sends to console
    ]
)

# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_logistic(model, input_data, seed, max_iter=1000):    
    X_train, y_train, X_val, y_val = input_data

    # Use LBFGS optimizer (same as sklearn's lbfgs)
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )
    
    criterion = nn.BCELoss()
    
    iteration = 0
    pbar = tqdm(total=max_iter, desc=f"Training {model.name} Seed={seed}", unit="iter", disable=True)

    def closure():
        nonlocal iteration
        model.train()
        optimizer.zero_grad()
        y_pred, _ = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        
        if iteration % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred, _ = model(X_val)
                y_test_pred = y_test_pred.cpu().numpy().flatten()
            test_ts = threat_score(y_val.cpu().numpy(), y_test_pred)
            model.train()
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Test TS': f'{test_ts:.4f}'})
        
        iteration += 1
        pbar.update(1)
        return loss
    
    # Run LBFGS optimization
    for epoch in range(max_iter):
        optimizer.step(closure)
        
        # Check for convergence
        if iteration >= max_iter:
            break
    
    pbar.close()    
    return model

def train_logistic_fast(model, X_train, y_train, X_test, y_test, seed, max_iter=1000):    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    pbar = tqdm(total=max_iter, desc=f"Training {model.name} Seed={seed}", disable=True)
    
    for iteration in range(max_iter):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if iteration % 50 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test).cpu().numpy().flatten()
            test_ts = threat_score(y_test.cpu().numpy(), y_test_pred)
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Test TS': f'{test_ts:.4f}'})
        
        pbar.update(1)
    
    pbar.close()
    return model

class ThreatScoreLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        # Use soft predictions for differentiability
        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1 - y_true))
        fn = torch.sum((1 - y_pred) * y_true)
        
        # Threat Score = TP / (TP + FP + FN)
        ts = tp / (tp + fp + fn + self.epsilon)
        
        # Return negative since we want to maximize TS
        return 1 - ts

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred, y_true):
        bce = nn.functional.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


def train_mamba(seed, model, input_data, max_epochs=200):
    X_train, y_train, X_val, y_val = input_data

    if model.spatial:
        X_train_spatial = create_spatial_tensors(X_train, features, k_neighbors=8)
        X_val_spatial = create_spatial_tensors(X_val, features, k_neighbors=8)
        X_train = (X_train, X_train_spatial)
        X_val = (X_val, X_val_spatial)

    model_save_path = f"./output/{model.name}_model.pth"
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    # Count class distribution

    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    criterion = nn.BCELoss()
    #criterion = FocalLoss(alpha=0.25, gamma=2.0)

    train_losses = []
    val_metrics = []
    best_ts = 0.0
        
    pbar = tqdm(range(max_epochs), desc=f"Training {model.name} Seed={seed}", unit="epoch", disable=True)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        y_pred, loss = model(X_train, y_train)
        #loss = criterion(y_pred, y_train)
        
        loss.backward()   
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_ts = threat_score(y_train.cpu().numpy(), y_pred.cpu().numpy().flatten())
            train_pred = (y_pred > 0.5).float()
            train_acc = accuracy_score(y_train.cpu().numpy(), train_pred.cpu().numpy())
            
            # Val metrics
            y_val_pred, val_loss = model(X_val, y_val)
            y_val_pred = y_val_pred.cpu().numpy().flatten()
            val_ts = threat_score(y_val.cpu().numpy(), y_val_pred)
            val_pred = (y_val_pred > 0.5).astype(int)
            val_acc = accuracy_score(y_val.cpu().numpy(), val_pred)
            val_f1 = f1_score(y_val.cpu().numpy(), val_pred)
            
        #scheduler.step(val_loss) # Update learning rate
        
        train_losses.append(loss.item())

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'TS': f'{val_ts:.4f}',
            'Acc': f'{val_acc:.4f}',
            'F1': f'{val_f1:.4f}',
            'Best': f'{best_ts:.4f}'
        })
        
        #if epoch % 10 == 0 or epoch == max_epochs - 1:
        #    tqdm.write(f"Epoch {epoch:3d}: Loss={loss.item():.6f}, Train Acc={train_acc:.4f}")
        #    tqdm.write(f"  Test TS={test_ts:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
        #    tqdm.write(f"  LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_ts > best_ts or best_ts == 0.0:
            best_ts = val_ts
            torch.save(model.state_dict(), model_save_path)
            
        val_metrics.append({
            'epoch': epoch,
            'ts': val_ts,
            'acc': val_acc,
            'f1': val_f1
        })
    
    pbar.close()
    model.load_state_dict(torch.load(model_save_path)) # Load best model
    os.remove(model_save_path)

    return model

def compare_all_approaches():
    data = PWFDF_Data()
    
    print(f"Total samples: {len(data.df)}")
    print(f"Training samples: {len(data.df[data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(data.df[data.df['Database'] == 'Test'])}\n")

    X_train_full, y_train_full, scaler = data.prepare_data_usgs(features, split='Training')
    #X_train, X_val, y_train, y_val = X_train_full, [], y_train_full, []
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)
    X_test, y_test, _ = data.prepare_data_usgs(features, split='Test', scaler=scaler)

    X_train = torch.Tensor(X_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    X_test = torch.Tensor(X_test).to(device)

    y_train = torch.Tensor(y_train).to(device)
    y_val = torch.Tensor(y_val).to(device)
    y_test = torch.Tensor(y_test).to(device)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")

    batch_size = 64
    input_dim = X_train.shape[-1]

    print(f"Batch Size: {batch_size}, Feature Size: {input_dim}")

    training_results = {}
    validation_results = {}
    test_results = {}
        
    best_seed_for_model = {}   # { model_name: seed }
    best_metrics_for_model = {}  # { model_name: metrics dict }
    best_ts_for_model = {}     # { model_name: TS float }
    worst_ts_for_model = {}     # { model_name: TS float }

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos

    model_classes = [
        #lambda: LogisticRegression(features, duration='15min'),
        #lambda: MambaClassifier(input_dim=input_dim, n_layers=2),
        #lambda: TSMixerClassifier(input_dim=input_dim),
        #lambda: InsaneMambaModel(features=features, duration='15min'),
        #lambda: StaticMLPModel(features=features, duration='15min'),
        #lambda: HybridMambaLogisticModelPro(features=features, pos_weight=pos_weight, d_model=64, n_layers=4, dropout=0.1),
        #lambda: MultiPathwayHybridModel(features=features, d_model=128, n_layers=6),
        #lambda: MultiBranchWithGlobalMamba()
        #lambda: Test(batch_size=batch_size, num_blocks=2),
    ]

    spatial_models = [
        lambda: Staley2017Model(features, duration='15min'),
        #lambda: SpatialMambaHybridModel(r_features, pos_weight),
        #lambda: SpatialMambaContextModel(r_features, pos_weight, 5 + 1)
        lambda: RandomForestModel(features, random_state=None),
        lambda: HybridMambaLogisticModel(features, pos_weight, input_dim=input_dim, n_layers=1),
        lambda: MultiPathwayHybridModel_og(features=features, pos_weight=pos_weight, d_model=128, n_layers=6),
        #lambda: HybridMambaFeatureGroups(features, pos_weight),
        lambda: HybridGroupedSpatialMambaModel(features=features, spatial_dim=16),
        lambda: HybridGroupedSpatialMambaModel2(features=features, spatial_dim=16),
    ]

    #seeds = [0, 10, 42, 51]
    seeds = range(0, 100)
    epochs = 100
    input_data = [X_train, y_train, X_val, y_val]

    logging.info("\n\n\n========================= Comparison =========================")
    logging.info(f"Total samples: {len(data.df)}")
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Val set: {X_val.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    logging.info(f"Input_data: {input_data[0].shape[0]}, {input_data[1].shape[0]}, {input_data[2].shape[0]}, {input_data[3].shape[0]}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Features Count: {len(features)}")
    logging.info(f"{features}")

    for seed in tqdm(seeds, desc="Seeds", position=0):
        for make_model in spatial_models:
            setup_seed(seed)
            model = make_model().to(device)

            model = model.to(device)

            if model.name == 'Staley' or model.name == 'LogisticRegression':
                model = train_logistic(model, input_data, seed, max_iter=epochs)
            elif model.name == 'RandomForest':
                model = train_random_forest(model, input_data, seed)
            else:
                model = train_mamba(seed, model, input_data, max_epochs=epochs)

            training_results[model.name] = evaluate_model(model, X_train, y_train)

            if len(X_val) > 0:
                validation_results[model.name] = evaluate_model(model, X_val, y_val)

            test_metrics = evaluate_model(model, X_test, y_test)
            test_results[model.name] = test_metrics

            ts = test_metrics['ts']
            name = model.name

            if name not in best_ts_for_model or ts > best_ts_for_model[name]:
                best_ts_for_model[name] = ts
                best_seed_for_model[name] = seed
                best_metrics_for_model[name] = test_metrics
                
                os.makedirs('./output/best_models', exist_ok=True)
                torch.save(model.state_dict(), f'./output/best_models/{name}_{seed}_best.pth')
            
            if name not in worst_ts_for_model or ts < worst_ts_for_model[name]:
                worst_ts_for_model[name] = ts

            torch.cuda.empty_cache()

        print_all_seeds = True
        if print_all_seeds:
            logging.info("\n" + "=" * 60)
            logging.info(f"SUMMARY (Seed = {seed})")
            logging.info("=" * 60)
            logging.info("Train set")
            for approach, results in training_results.items():
                logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f}")
            logging.info("=" * 60)
            logging.info("Test set")
            for approach, results in test_results.items():
                logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")

    logging.info("\n========================= BEST SEEDS =========================")
    for name in best_seed_for_model:
        logging.info(f"\nModel: {name}")
        logging.info(f"  Best Seed:  {best_seed_for_model[name]}")
        logging.info(f"  Best TS:    {best_ts_for_model[name]:.4f}")
        logging.info(f"  Best Worst: {worst_ts_for_model[name]:.4f}")
        logging.info(f"  Metrics:    {best_metrics_for_model[name]}")


if __name__ == "__main__":
    setup_seed(42)
    compare_all_approaches()
