import torch
import torch.nn as nn
import torch.nn.functional as F


# BASE MODULE =============================

class MraAuxiliaryModule:
    """
    Base class for plug-and-play auxiliary modules in a modular RL agent.
    
    Features:
        - Each module can define its own prediction head (e.g., nn.Module).
        - Supports multiple auxiliary tasks (e.g., classification, regression).
        - Plug-in system: implement a new auxiliary task by subclassing.
        - Provides hooks for loss, metrics, and head definition.
        - Integrates cleanly with agent logging/optimization code.
    """
    def __init__(self, name, task="classification"):
        """
        Args:
            name (str): Name identifier for this auxiliary module (e.g., "cue", "confidence").
            task (str): Auxiliary task type, e.g. "classification" or "regression".
        """
        self.name = name
        self.head = None  # To be set by subclasses: an nn.Module for prediction
        self.task = task

    def aux_loss(self, pred, target, context=None):
        """
        Computes the auxiliary loss for this task.
        Should be implemented by the subclass.

        Args:
            pred (torch.Tensor): Model predictions from the auxiliary head.
            target (torch.Tensor): Ground-truth targets.
            context (dict, optional): Additional context for loss (unused by default).

        Returns:
            torch.Tensor: Loss value (scalar).
        """
        raise NotImplementedError

    def get_head(self, feat_dim):
        """
        Returns the prediction head (an nn.Module), given the input feature dimension.
        Subclasses should override this to define their specific head.

        Args:
            feat_dim (int): Input feature dimension for the head.

        Returns:
            nn.Module: The prediction head.
        """
        raise NotImplementedError

    def aux_metrics(self, pred, target, context=None):
        """
        Computes diagnostic metrics for evaluation and logging.
        By default, uses task type to choose metric.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.
            context (dict, optional): Additional info for metrics (unused by default).

        Returns:
            dict: Dictionary of metric names and values (e.g., {"acc": 0.93}).
        """
        if self.task == "classification":
            return self.__classification_metric(pred, target, context)
        else:
            return self.__regression_metric(pred, target, context)

    def __classification_metric(self, pred, target, context=None):
        """
        Computes accuracy for classification tasks.

        Args:
            pred (torch.Tensor): Model logits (N x C).
            target (torch.Tensor): Integer class labels (N,).

        Returns:
            dict: {"acc": float}
        """
        pred_label = pred.argmax(dim=-1)
        correct = (pred_label == target).float()
        acc = correct.mean().item()
        return {'acc': acc}

    def __regression_metric(self, pred, target, context=None):
        """
        Computes mean squared error for regression tasks.

        Args:
            pred (torch.Tensor): Model predictions (N x 1 or N).
            target (torch.Tensor): Ground-truth values (N,).

        Returns:
            dict: {"mse": float}
        """
        mse = F.mse_loss(torch.sigmoid(pred.squeeze(-1)), target.float()).item()
        return {'mse': mse}


# AUXILIAR PLUGINS ========================

class CueAuxModule(MraAuxiliaryModule):
    """
    Auxiliary module for classification of a "cue" signal.
    Predicts cue class using a linear head and cross-entropy loss.
    """
    def __init__(self, feat_dim, n_classes=2):
        """
        Args:
            feat_dim (int): Input feature dimension.
            n_classes (int): Number of classes for cue prediction.
        """
        super().__init__("cue", task="classification")
        self.head = nn.Linear(feat_dim, n_classes)

    def aux_loss(self, pred, target, context=None):
        """
        Computes classification loss (cross-entropy).

        Args:
            pred (torch.Tensor): Logits from cue head (N x n_classes).
            target (torch.Tensor): Integer class labels (N,).

        Returns:
            torch.Tensor: Scalar loss.
        """
        return F.cross_entropy(pred, target)


class ConfidenceAuxModule(MraAuxiliaryModule):
    """
    Auxiliary module for confidence regression.
    Predicts confidence as a scalar using a linear head and MSE loss.
    """
    def __init__(self, feat_dim):
        """
        Args:
            feat_dim (int): Input feature dimension.
        """
        super().__init__("confidence", task="regression")
        self.head = nn.Linear(feat_dim, 1)

    def aux_loss(self, pred, target, context=None):
        """
        Computes regression loss (MSE).

        Args:
            pred (torch.Tensor): Confidence predictions (N x 1).
            target (torch.Tensor): Target values (N,).

        Returns:
            torch.Tensor: Scalar loss.
        """
        return F.mse_loss(torch.sigmoid(pred.squeeze(-1)), target.float())

    def aux_metrics(self, pred, target, context=None):
        """
        Computes mean squared error (MSE) for confidence regression.

        Args:
            pred (torch.Tensor): Predictions (N x 1).
            target (torch.Tensor): Targets (N,).

        Returns:
            dict: {"mse": float}
        """
        mse = F.mse_loss(torch.sigmoid(pred.squeeze(-1)), target.float()).item()
        return {'mse': mse}
