import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix

from eval import threat_score

# Gets very similar values, so they masked out the data in a similar way to how it is done in prepare data

class Staley2017Model(nn.Module):
    """
    Staley et al. (2017) logistic regression model:
    p = 1 / (1 + exp(-(B + Ct*T*R + Cf*F*R + Cs*S*R)))
    """
    
    def __init__(self):
        super().__init__()
        # Initialize all parameters at 0
        self.B = nn.Parameter(torch.tensor([0.0]))
        self.Ct = nn.Parameter(torch.tensor([0.0]))
        self.Cf = nn.Parameter(torch.tensor([0.0]))
        self.Cs = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, T, F, S, R):
        B = self.B.squeeze()
        Ct = self.Ct.squeeze()
        Cf = self.Cf.squeeze()
        Cs = self.Cs.squeeze()
        
        logit = B + Ct * T * R + Cf * F * R + Cs * S * R
        # Clip for numerical stability
        #logit = torch.clamp(logit, -500, 500)
        return torch.sigmoid(logit).unsqueeze(1)


def prepare_data(pwfdf_data, duration='15min', split='Training'):
    """
    Prepare training data directly from PWFDF_Data using Database column
    
    Args:
        pwfdf_data: PWFDF_Data object
        duration: '15min', '30min', or '60min'
        split: 'Training' or 'Test'
    """
    df = pwfdf_data.df
    
    # Filter by Database column
    df = df[df['Database'] == split].copy()
    
    # Extract features
    T = df['PropHM23'].values
    F = df['dNBR/1000'].values
    S = df['KF'].values
    
    # Get rainfall based on duration
    if duration == '15min':
        R = df['Acc015_mm'].values
    elif duration == '30min':
        R = df['Acc030_mm'].values
    else:  # 60min
        R = df['Acc060_mm'].values
    
    y = df['Response'].values
    
    # Remove rows with missing values
    mask = ~(np.isnan(T) | np.isnan(F) | np.isnan(S) | np.isnan(R))
    #mask = ~np.isnan(X).any(axis=1)
    T, F, S, R, y = T[mask], F[mask], S[mask], R[mask], y[mask]
    
    #T = np.nan_to_num(T, nan=0.0)
    #F = np.nan_to_num(F, nan=0.0)
    #S = np.nan_to_num(S, nan=0.0)
    #R = np.nan_to_num(R, nan=0.0)

    return T, F, S, R, y


def train(model, pwfdf_data, duration='15min', max_iter=1000):
    """Train the model using LBFGS optimizer"""
    # Prepare training data
    T_train, F_train, S_train, R_train, y_train = prepare_data(pwfdf_data, duration, split='Training')
    T_test, F_test, S_test, R_test, y_test = prepare_data(pwfdf_data, duration, split='Test')
    
    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {np.sum(y_train)} ({100*np.mean(y_train):.1f}%)")
    print(f"Negative samples: {len(y_train) - np.sum(y_train)} ({100*(1-np.mean(y_train)):.1f}%)")
    print(f"Testing on {len(y_test)} samples")

    # Convert to tensors with double precision for better numerical stability
    T_train_t = torch.DoubleTensor(T_train)
    F_train_t = torch.DoubleTensor(F_train)
    S_train_t = torch.DoubleTensor(S_train)
    R_train_t = torch.DoubleTensor(R_train)
    y_train_t = torch.DoubleTensor(y_train).reshape(-1, 1)
    
    T_test_t = torch.DoubleTensor(T_test)
    F_test_t = torch.DoubleTensor(F_test)
    S_test_t = torch.DoubleTensor(S_test)
    R_test_t = torch.DoubleTensor(R_test)
    
    # Convert model to double precision
    model = model.double()
    
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
    print(f"Training {duration} model with LBFGS...")
    
    iteration = 0
    
    def closure():
        nonlocal iteration
        optimizer.zero_grad()
        y_pred = model(T_train_t, F_train_t, S_train_t, R_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        
        if iteration % 10 == 0:
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_test_pred = model(T_test_t, F_test_t, S_test_t, R_test_t).numpy().flatten()
            test_ts = threat_score(y_test, y_test_pred)
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
    
    print(f"\n{'='*50}")
    print(f"Training Set Results for {duration}")
    evaluate(model, T_train, F_train, S_train, R_train, y_train, duration=duration)

    return model, (T_test, F_test, S_test, R_test, y_test)


def evaluate(model, T_test, F_test, S_test, R_test, y_test, duration='15min', threshold=0.5):
    """Evaluate model with fixed threshold"""
    T_test_t = torch.DoubleTensor(T_test)
    F_test_t = torch.DoubleTensor(F_test)
    S_test_t = torch.DoubleTensor(S_test)
    R_test_t = torch.DoubleTensor(R_test)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(T_test_t, F_test_t, S_test_t, R_test_t).numpy().flatten()
    
    # Use fixed threshold instead of optimizing
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary, labels=[0, 1]).ravel()
    
    ts = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{'='*50}")
    print(f"Threat Score: {ts:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"Threshold:    {threshold:.3f} (fixed)")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"\nLearned Parameters:")
    print(f"  B  = {model.B.item():.4f}")
    print(f"  Ct = {model.Ct.item():.4f}")
    print(f"  Cf = {model.Cf.item():.4f}")
    print(f"  Cs = {model.Cs.item():.4f}")
    
    return ts


def compare_params(models, durations):
    """Compare learned vs published Staley 2017 parameters"""
    published = {
        '15min': {'B': -3.63, 'Ct': 0.41, 'Cf': 0.67, 'Cs': 0.70},
        '30min': {'B': -3.61, 'Ct': 0.26, 'Cf': 0.39, 'Cs': 0.50},
        '60min': {'B': -3.21, 'Ct': 0.17, 'Cf': 0.20, 'Cs': 0.22}
    }
    
    print(f"\n{'='*60}")
    print("Comparison with Staley 2017 Published Parameters")
    print(f"{'='*60}")
    
    for dur in durations:
        print(f"\n{dur}:")
        print(f"{'Param':<8} {'Published':<12} {'Learned':<12} {'Diff':<12}")
        print("-" * 50)
        
        for param in ['B', 'Ct', 'Cf', 'Cs']:
            pub = published[dur][param]
            learn = models[dur].state_dict()[param].item()
            diff = learn - pub
            print(f"{param:<8} {pub:<12.4f} {learn:<12.4f} {diff:<12.4f}")

# Main execution
if __name__ == "__main__":
    from data import PWFDF_Data
    
    # Load data
    print("Loading PWFDF data...")
    pwfdf = PWFDF_Data()
    
    print(f"Total samples: {len(pwfdf.df)}")
    print(f"Training samples: {len(pwfdf.df[pwfdf.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(pwfdf.df[pwfdf.df['Database'] == 'Test'])}\n")
    
    # Train models for each duration
    durations = ['15min', '30min', '60min']
    models = {}
    test_data = {}
    
    for duration in durations:
        print(f"\n{'='*60}")
        print(f"Training {duration} model")
        print(f"{'='*60}")
        
        model = Staley2017Model()
        model, test_data[duration] = train(model, pwfdf, duration=duration, max_iter=100)
        models[duration] = model
        
        T_test, F_test, S_test, R_test, y_test = test_data[duration]
        print(f"\n{'='*50}")
        print(f"Test Set Results for {duration}")
        evaluate(model, T_test, F_test, S_test, R_test, y_test, duration=duration)
    
    # Compare with published parameters
    compare_params(models, durations)