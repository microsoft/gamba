from typing import Optional, Tuple

import torch.nn as nn
import torch
import torch.nn.functional as F


class OAMaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, reweight: bool = True):
        super().__init__()
        self.reweight = reweight
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        timesteps: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked cross-entropy loss for sequences. Evaluates the cross-entropy loss at specified locations in a
        sequence. When reweight = True, reweights CE according to Hoogeboom et al.; reweight term = 1/(D-t+1).

        Parameters:
        -----------
        pred: torch.Tensor (any fp type)
            Predictions from the model (N, L, n_tokens)
        tgt: torch.Tensor (torch.long)
            Target values (N, L)
        mask: torch.Tensor (torch.bool)
            True where the masking token was applied (N, L)
        timesteps: torch.Tensor (torch.long)
            Number of masked tokens in the sequence (N,)
        input_mask: torch.Tensor (torch.bool)
            True where the tokens are from a sequence rather than padding (N, L)
        """
        input_mask = input_mask.bool()
        nonpad_tokens = input_mask.sum(dim=1)

        # we only want to compute the error over the masked tokens
        # this also eliminates the contribution of padding tokens since they aren't in the mask (by construction)
        tgt = tgt * mask + ~mask * -100

        loss = F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]),
            tgt.flatten(),
            weight=self.weight,
            reduction="none",
        ).reshape(*tgt.shape)
        nll_loss = loss.sum()

        if self.reweight:
            rwt_term = 1.0 / timesteps
            rwt_term = rwt_term[:, None]
            _n_tokens = nonpad_tokens[:, None]
            ce_loss = (_n_tokens * rwt_term * loss).sum()
        else:
            ce_loss = nll_loss
        return ce_loss, nll_loss


class MaskedCrossEntropyLoss(nn.Module):
    """Masked cross-entropy loss for sequences. Evalutes the CE where the mask is True."""

    def __init__(self, weight=None, reduction="mean"):
        """Creates a MaskedCrossEntropyLoss module.

        Parameters:
        -----------
        weight: torch.Tensor
            Weights for the CE loss. Default is uniform.
        reduction: str
            How to reduce the loss. Default is "mean".

        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # we only want to compute the error over the masked tokens
        # this also eliminates the contribution of padding tokens since they aren't in the mask (by construction)
        tgt = tgt * mask + (1 - mask) * -100

        return F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]),
            tgt.flatten(),
            weight=self.weight,
            reduction=self.reduction,
        )


class GaussianNLLLoss(nn.Module):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)

    def forward(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:

        # let's return the loss as the negative log likelihood of the target given the predicted parameters of the Gaussian distribution
        # where pred: torch.Tensor has shape (batch, seq_length, 2) where 2 is the mean and variance of the Gaussian distribution
        # we will use the - log likelihood of the Gaussian distribution as the loss

        # mask is where tgt is not equal to -100
        mask = tgt != -100

        mean = pred[:, :, 0]
        log_var = pred[:, :, 1]

        # apply the mask to mean, log_var and tgt
        mean = mean[mask]
        log_var = log_var[mask]
        tgt = tgt[mask]

        # exponentiate variance and mean
        var = torch.exp(log_var)
        mean = torch.exp(mean)

        #print the mean and variance
        print(f"mean: {mean}")
        print(f"var: {var}")

        # loss using PyTorch's built-in GaussianNLLLoss
        loss = self.loss_fn(mean, tgt, var)
        
        return loss

class InverseGammaNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # mask is where tgt is not equal to -100
        mask = tgt != -100

        # let's return the loss as the negative log likelihood of the target given the predicted parameters of the inverse Gamma distribution
        # where pred: torch.Tensor has shape (batch, seq_length, 2) where 2 is the scaling parameter theta and shape parameter k of the inverse gamma distribution
        # we will use the - log likelihood of the inverse gamma distribution as the loss
        log_scaling = pred[:, :, 0]
        log_shape = pred[:, :, 1]

        # apply the mask to log_scaling, log_shape and tgt
        log_scaling = log_scaling[mask]
        log_shape = log_shape[mask]
        tgt = tgt[mask]
        print(f"in inverse gamma loss tgt: {tgt}")

        # exponentiate scaling and shape
        scaling = torch.exp(log_scaling)
        shape = torch.exp(log_shape)

        print(f"in inverse gamma loss scaling and shape, {scaling}, {shape}")

        # pytorch distribution is more stable
        inv_gamma_dist = torch.distributions.inverse_gamma.InverseGamma(shape, scaling)
        log_pdf = inv_gamma_dist.log_prob(tgt)
        print("LOSS: ", -log_pdf)
        loss = -log_pdf

        # mean loss over batch and seq length
        loss = loss.mean()
        return loss


class PoissonNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        print("pred shape: ", pred.shape)
        print("tgt shape: ", tgt.shape)
        # mask is where tgt is not equal to -100
        mask = tgt != -100

        # let's return the loss as the negative log likelihood of the target given the predicted parameters of the poisson distribution
        # where pred: torch.Tensor has shape (batch, seq_length, 1) where this represents lambda param
        # we will use the - log likelihood of the poisson distribution as the loss
        log_lam = pred

        # apply the mask to log_scaling, log_shape and tgt
        log_lam = log_lam[mask]
        tgt = tgt[mask]
        print(f"in poisson loss tgt: {tgt}")

        # exponentiate lambda
        lam = torch.exp(log_lam)

        print(f"in poisson loss lambda, {lam}")

        # pytorch distribution is more stable
        poisson_dist = torch.distributions.poisson.Poisson(lam)
        log_pdf = poisson_dist.log_prob(tgt)
        print("LOSS: ", -log_pdf)
        loss = -log_pdf

        # mean loss over batch and seq length
        loss = loss.mean()
        return loss


# use pytorch implementation
# log the gradients of the loss
# clip the gradients
# see if other distributions are more stable
# error correlated with the presence of species at sites
