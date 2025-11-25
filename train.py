import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data import PWFDF_Data
from eval import find_best_threshold, evaluate, compare_params
from models.log_reg import Staley2017Model
from models.mamba import MambaClassifier, HybridMambaLogisticModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_lg(model, X_train, y_train, X_test, y_test, max_iter=1000):    
    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")
    
    y_train = y_train.reshape(-1, 1)
    
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
    
    # Training with LBFGS
    print(f"Training {model.duration} model with LBFGS...")
    
    iteration = 0
    
    def closure():
        nonlocal iteration
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        
        if iteration % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test).cpu().numpy().flatten()
            threshold, test_ts = find_best_threshold(y_test.cpu().numpy(), y_test_pred)
            model.train()
            
            print(f"Iter {iteration}: Loss={loss.item():.6f}, Test TS={test_ts:.4f}")
            print(f"  B={model.B.item():.4f}, Ct={model.Ct.item():.4f}, "
                  f"Cf={model.Cf.item():.4f}, Cs={model.Cs.item():.4f}")
        
        iteration += 1
        return loss
    
    # Run LBFGS optimization
    for epoch in range(max_iter):
        optimizer.step(closure)
        
        # Check for convergence
        if iteration >= max_iter:
            break
    
    print(f"Training completed after {iteration} iterations")
    
    return model

def train_mamba(model, X_train, y_train, X_test, y_test, max_epochs=100, patience=10):    
    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()
    
    train_losses = []
    test_metrics = []
    best_ts = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("Training Mamba model with AdamW optimizer...")
    
    model_save_path = './output/best_mamba_model.pth'

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_pred = (y_pred > 0.5).float()
            train_acc = accuracy_score(y_train.cpu().numpy(), train_pred.cpu().numpy())
            
            # Test metrics
            y_test_pred = model(X_test).cpu().numpy().flatten()
            threshold, test_ts = find_best_threshold(y_test.cpu().numpy(), y_test_pred)
            test_pred = (y_test_pred > threshold).astype(int)
            test_acc = accuracy_score(y_test.cpu().numpy(), test_pred)
            test_f1 = f1_score(y_test.cpu().numpy(), test_pred)
            
        # Update learning rate
        scheduler.step(test_ts)
        
        # Print progress
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.6f}, Train Acc={train_acc:.4f}")
            print(f"  Test TS={test_ts:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
            print(f"  LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if test_ts > best_ts:
            best_ts = test_ts
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        train_losses.append(loss.item())
        test_metrics.append({
            'epoch': epoch,
            'ts': test_ts,
            'acc': test_acc,
            'f1': test_f1
        })
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    print(f"Training completed. Best Test TS={best_ts:.4f} at epoch {best_epoch}")
    
    return model

def main():
    data = PWFDF_Data()
    
    print(f"Total samples: {len(data.df)}")
    print(f"Training samples: {len(data.df[data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(data.df[data.df['Database'] == 'Test'])}\n")
    
    X_train, y_train = data.prepare_data_with_normalization(split='Training')
    X_test, y_test = data.prepare_data_with_normalization(split='Test')

    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    #model = Staley2017Model().to(device)
    model = MambaClassifier(input_dim=X_train.shape[1]).to(device)
    #model = HybridMambaLogisticModel(input_dim=X_train.shape[1]).to(device)

    #model = train_lg(model, X_train, y_train, X_test, y_test, max_iter=100)
    model = train_mamba(model, X_train, y_train, X_test, y_test, max_epochs=100, patience=15)
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()