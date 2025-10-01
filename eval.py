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
  cm = confusion_matrix(y_true, y_pred)
  tn, fp, fn, tp = cm.ravel()

  # Threat Score (CSI)
  threat_score = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

  return threat_score, {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

def evaluate(model, loader, return_probs=False):
  model.eval()
  all_labels = []
  all_predictions = []
  all_probabilities = []

  with torch.no_grad():
    for features, labels in loader:
      outputs, loss = model(features, labels)
      probabilities = F.softmax(outputs, dim=1)
      _, predicted = torch.max(outputs.data, 1)

      all_labels.extend(labels.cpu().numpy())
      all_predictions.extend(predicted.cpu().numpy())
      all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1

  all_labels = np.array(all_labels)
  all_predictions = np.array(all_predictions)
  all_probabilities = np.array(all_probabilities)

  # Calculate metrics
  accuracy = np.mean(all_predictions == all_labels)
  threat_score, confusion = calculate_threat_score(all_labels, all_predictions)
  auc = roc_auc_score(all_labels, all_probabilities)

  if return_probs:
      return accuracy, threat_score, auc, confusion, all_labels, all_probabilities

  return accuracy, threat_score, auc, confusion
