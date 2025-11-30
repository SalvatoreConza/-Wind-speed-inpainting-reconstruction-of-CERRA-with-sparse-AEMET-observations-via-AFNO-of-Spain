# #  SALVATORE MAIN MODIFICATIONS: 
# - just added the HoleLoss class, ignore the MaskedMeanSquaredLoss class

# script that define several loss functions MSELoss, RMSELoss,
# LogRMSELoss, LogMSELoss, and a controller class GeneralPM25Loss
import torch
import torch.nn as nn
# defining several loss functions from PyTorch nn.Module which 
# is a standard PyTorch way of building models or losses

# MSE Loss: Mean Squared Error, is the classic one for regression
# difficult to interpret
class MSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

# RMSE Loss: Root Mean Squared Error, is the square root of MSE
# easy to interpret since have the same unit as original data
class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# LogRMSE Loss: Root Mean Squared Error of the logarithm of predictions and true values
# LogMSE Loss: Mean Squared Error of the logarithm of predictions and true values
# these two losses both mesure the relative error(or percentage error) between prediction and true values
class LogRMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        log_pred = torch.log(y_pred + self.eps)
        log_true = torch.log(y_true + self.eps)
        return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, mask):
        log_pred = torch.log(y_pred + self.eps)
        log_true = torch.log(y_true + self.eps)
        return torch.mean((log_pred - log_true) ** 2)
        
class MaskedMeanSquaredLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

class HoleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    def forward(self, y_pred, y_true, mask):
        print('this loss is being used')
        if isinstance(mask, dict):
            print(f"Mask keys: {mask.keys()}")
        return self.l1((1 - mask) * y_pred, (1 - mask) * y_true)

# smooth L1 loss, compromise beetween L1 and L2 loss
# it's roboust to outliers like L1 loss because it threats large errors more gently respect to L2
# differentiable everywhere like L2 loss
class CharbonnierLoss(nn.Module):
    """
    Smooth L1 approximation: L = mean( sqrt((y_pred - y_true)^2 + eps^2) )
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        # add small eps^2 for numerical stability
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

# GeneralPM25Loss: controller class that can select and combine different loss functions
class GeneralPM25Loss(nn.Module):
    def __init__(self, loss_type='rmse', log_weight=0.5, eps=1e-6):
        """
        Parameters:
        - loss_type: 'mse', 'rmse', 'logrmse', 'logmse', 'charbonnier' (or 'charb'), or 'combined'
        - log_weight: weight of log-based loss in combined mode
        - eps: small value to avoid log(0) and stabilize Charbonnier
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps
        self.log_weight = log_weight

        valid = {'mse', 'rmse', 'logrmse', 'maskedmeansquaredloss', 'holeloss', 'logmse', 'combined', 'charbonnier', 'charb'}
        if self.loss_type not in valid:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        if self.loss_type == 'mse':
            self.mse = MSELoss()
            self.forward = self._mse_forward

        elif self.loss_type == 'rmse':
            self.rmse = RMSELoss()
            self.forward = self._rmse_forward

        elif self.loss_type == 'logrmse':
            self.logrmse = LogRMSELoss(eps=eps)
            self.forward = self._logrmse_forward

        elif self.loss_type == 'logmse':
            self.logmse = LogMSELoss(eps=eps)
            self.forward = self._logmse_forward
        
        elif self.loss_type == 'maskedmeansquaredloss':
            self.maskedmeansquaredloss = MaskedMeanSquaredLoss()
            self.forward = self._maskedmeansquaredloss_forward

        elif self.loss_type == 'holeloss':
            self.holeloss = HoleLoss()
            self.forward = self._holeloss_forward
            
        elif self.loss_type in {'charbonnier', 'charb'}:
            self.charb = CharbonnierLoss(eps=eps)
            self.forward = self._charbonnier_forward

        elif self.loss_type == 'combined':
            self.rmse = RMSELoss()
            self.logrmse = LogRMSELoss(eps=eps)
            self.forward = self._combined_forward
    # wrapper functions that take y_pred and y_true and pass them
    # to the actual loss objects:
    def _mse_forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)

    def _rmse_forward(self, y_pred, y_true):
        return self.rmse(y_pred, y_true)

    def _logrmse_forward(self, y_pred, y_true):
        return self.logrmse(y_pred, y_true)

    def _logmse_forward(self, y_pred, y_true):
        return self.logmse(y_pred, y_true)

    def _maskedmeansquaredloss_forward(self,y_pred,y_true):
        return self.maskedmeansquaredloss(y_pred,y_true)

    def _holeloss_forward(self,y_pred,y_true,mask):
        return self.holeloss(y_pred,y_true,mask)
        
    def _charbonnier_forward(self, y_pred, y_true):
        return self.charb(y_pred, y_true)
    # special case, hybrid weighert average loss of RMSE and LogRMSE:
    def _combined_forward(self, y_pred, y_true):
        return (
            (1 - self.log_weight) * self.rmse(y_pred, y_true)
            + self.log_weight * self.logrmse(y_pred, y_true)
        )
