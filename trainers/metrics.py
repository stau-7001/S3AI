from typing import Union

import torch
# from ogb.graphproppred import Evaluator
# from ogb.lsc import PCQM4MEvaluator
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import f1_score,average_precision_score,auc,precision_recall_curve, recall_score, accuracy_score, roc_auc_score, log_loss, matthews_corrcoef
# from datasets.geom_drugs_dataset import GEOMDrugs
# from datasets.qm9_dataset import QM9Dataset
class F1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        try:
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets_indices = torch.argmax(targets, dim=1)
                pred_classes = torch.argmax(preds, dim=1)
            else:
                pred_classes = np.around(preds.squeeze().cpu())
                targets_indices = targets.squeeze().cpu()

            score = f1_score(targets_indices.cpu(), pred_classes.cpu())
            return score
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, handle the error in a specific way, 
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0
    
def mcc_score(predict_proba, label):
    trans_pred = np.ones(predict_proba.shape)
    trans_label = np.ones(label.shape)
    trans_pred[predict_proba < 0.5] = -1
    trans_label[label != 1] = -1
    # print(trans_pred.shape, trans_pred)
    # print(trans_label.shape, trans_label)
    mcc = matthews_corrcoef(trans_label, trans_pred)
    # mcc = metricser.matthews_corrcoef(trans_pred, trans_label)
    return mcc

class MCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        try:
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets_indices = torch.argmax(targets, dim=1)
                pred_classes = torch.argmax(preds, dim=1)
            else:
                score = mcc_score(preds.squeeze().cpu(), targets.squeeze().cpu())
                return score
            score = matthews_corrcoef(targets_indices.cpu(), pred_classes.cpu())
            return score
        except Exception as e:
            print(f"An error occurred: {e}")
            # print(targets_indices.shape,pred_classes.shape)
            # Optionally, handle the error in a specific way, 
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0
    
class ROCAUC(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets_indices = torch.argmax(targets, dim=1)
            mask = targets == 1
            pred_score = preds[mask]
            pred_score = pred_score.view(-1)
        else:
            pred_score = preds.squeeze().view(-1)
            targets_indices = targets.squeeze().view(-1)
        # print(targets_indices,pred_score)
        score = 1.
        try:
            score = roc_auc_score(targets_indices.cpu(), pred_score.cpu())
            return score
        except ValueError:
            pass
        return score
    
class PRAUC(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets_indices = torch.argmax(targets, dim=1).cpu()
            mask = targets == 1
            pred_score = preds[mask]
            pred_score = pred_score.view(-1).cpu()
        else:
            pred_score = preds.squeeze().view(-1).cpu()
            targets_indices = targets.squeeze().view(-1).cpu()
        # print(targets_indices,pred_score)
        score = 1.
        try:
            precision, recall, threshold = precision_recall_curve(y_true=targets_indices, probas_pred=pred_score)
            score = auc(recall, precision)
            return score
        except ValueError:
            pass
        return score
    
class PearsonR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        try:
            shifted_x = preds - torch.mean(preds, dim=0)
            shifted_y = targets - torch.mean(targets, dim=0)
            sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
            sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

            pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1e-8)
            pearson = torch.clamp(pearson, min=-1, max=1)
            pearson = pearson.mean()
            return pearson
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, handle the error in a specific way, 
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0

class MAE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        loss = F.l1_loss(preds, targets)
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        mse = torch.mean((preds - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse

class ClassificationAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):     
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)

        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets_indices = torch.argmax(targets, dim=1).cpu()
            pred_classes = torch.argmax(preds, dim=1).cpu()
        else:
            pred_classes = np.around(preds.squeeze().view(-1).cpu())
            targets_indices = targets.squeeze().view(-1).cpu()
        return accuracy_score(y_true=targets_indices, y_pred=pred_classes)
        

class SpearmanR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
#         print(preds,targets)
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        preds_rank = torch.argsort(torch.argsort(preds, dim=0), dim=0).float()
        targets_rank = torch.argsort(torch.argsort(targets, dim=0), dim=0).float()

        shifted_x = preds_rank - torch.mean(preds_rank, dim=0)
        shifted_y = targets_rank - torch.mean(targets_rank, dim=0)
        sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
        sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

        spearman = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1e-8)
        spearman = torch.clamp(spearman, min=-1, max=1)
        spearman = spearman.mean()
        return spearman

class SpearmanCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = preds.detach().numpy().reshape(-1)
        targets = targets.detach().numpy().reshape(-1)
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        return stats.spearmanr(preds, targets)[0]



def denormalize(normalized: torch.tensor, means, stds, eV2meV):
    denormalized = normalized * stds[None, :] + means[None, :]  # [batchsize, n_tasks]
    if eV2meV:
        denormalized = denormalized * eV2meV[None, :]
    return denormalized