import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def calculate_threat_score(y_true, y_pred):
  """
  Calculate Threat Score (Critical Success Index)
  TS = hits / (hits + misses + false alarms)
  """
  cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
  tn, fp, fn, tp = cm.ravel()

  # Threat Score (CSI)
  threat_score = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

  return threat_score, {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

class Evaluation_Metrics:
  accuracy: float
  loss: float

  threat_score: float
  auc: float
  confusion: dict

  def __init__(self) -> None:
    pass

  def output(self):
      print(f"Acc: {self.accuracy:.4f} | Loss: {self.loss:.4f} | Threat: {self.threat_score:.4f} | AUC: {self.auc:.4f}")
      print("Confusion Matrix:", self.confusion)

def evaluate(model, loader, return_probs=False):
    out = Evaluation_Metrics()
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    losses = torch.zeros(len(loader))
    
    with torch.no_grad():
        for i, (features, labels) in enumerate(loader):
            outputs, loss = model(features, labels)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            losses[i] = loss.item()
            all_labels.append(labels)
            all_predictions.append(predicted)
            all_probabilities.append(probabilities[:, 1])  # Probability of class 1
    
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    all_probabilities = torch.cat(all_probabilities)
    
    # Calculate metrics
    out.accuracy = (all_predictions == all_labels).float().mean().item()
    out.threat_score, out.confusion = calculate_threat_score(all_labels, all_predictions)
    out.auc = roc_auc_score(all_labels.cpu().numpy(), all_probabilities.cpu().numpy())
    out.loss = losses.mean().item()
    
    if return_probs:
        return out, all_labels, all_probabilities
    return out