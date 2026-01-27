"""
Jensen-Shannon Divergence Loss Module
======================================
This module implements the Jensen-Shannon Divergence (JSD) loss function
used to train neural networks to match the GP oracle's probability distributions.

JSD is a symmetric measure of similarity between two probability distributions,
ensuring that predicted haptic signals statistically match the GP's estimates.

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"

From the paper (Section 4.1):
- JSD(P_GP || Q_NN) = 0.5 * KL(P_GP || M) + 0.5 * KL(Q_NN || M)
- M = 0.5 * (P_GP + Q_NN) is the mixture distribution
- Since M is a two-component Gaussian mixture, no closed-form solution exists
- We therefore employ a Monte Carlo approximation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class JSDLoss(nn.Module):
    """
    Jensen-Shannon Divergence Loss Function with Monte Carlo Approximation.
    
    The JSD loss measures the statistical similarity between the neural network's
    predicted distribution (Q_NN) and the GP oracle's distribution (P_GP).
    
    From the paper (Eq. 5):
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q) is the mixture distribution.
    
    Since M is a two-component Gaussian mixture rather than a single Gaussian,
    no closed-form solution exists. We therefore employ Monte Carlo approximation.
    
    From the paper (Eq. 7):
    KL(P_GP || M) ≈ (1/L) * Σ [log p(x^(l)) - log m(x^(l))]
    where m(x) = 0.5 * p(x) + 0.5 * q(x)
    
    Properties:
    - Symmetric: JSD(P||Q) = JSD(Q||P)
    - Bounded: 0 ≤ JSD ≤ log(2)
    - Zero iff P = Q (distributions are identical)
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        reduction: str = 'mean',
        epsilon: float = 1e-10
    ):
        """
        Initialize JSD Loss with Monte Carlo approximation.
        
        Args:
            n_samples: Number of Monte Carlo samples (L in paper)
            reduction: How to reduce batch losses ('mean', 'sum', or 'none')
            epsilon: Small constant for numerical stability
        """
        super(JSDLoss, self).__init__()
        
        # Store configuration
        self.n_samples = n_samples  # L in paper
        self.reduction = reduction
        self.epsilon = epsilon  # Prevent log(0) errors
    
    def gaussian_log_prob(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability density of Gaussian distribution.
        
        log p(x) = -0.5 * [k*log(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ)]
        
        For diagonal covariance (independent dimensions):
        log p(x) = -0.5 * Σ_i [log(2π) + log(σ_i²) + ((x_i - μ_i) / σ_i)²]
        
        Args:
            x: Sample points, shape (batch_size, n_samples, output_dim)
            mean: Distribution mean, shape (batch_size, 1, output_dim)
            std: Distribution std, shape (batch_size, 1, output_dim)
            
        Returns:
            Log probabilities, shape (batch_size, n_samples)
        """
        # Ensure numerical stability
        std = std + self.epsilon
        var = std ** 2
        
        # Compute log probability for each dimension
        # log p(x_i) = -0.5 * [log(2π) + log(σ²) + ((x - μ)/σ)²]
        log_prob_per_dim = -0.5 * (
            np.log(2 * np.pi) + 
            torch.log(var) + 
            ((x - mean) ** 2) / var
        )
        
        # Sum over dimensions (assuming independence)
        # Shape: (batch_size, n_samples)
        log_prob = torch.sum(log_prob_per_dim, dim=-1)
        
        return log_prob
    
    def mixture_log_prob(
        self,
        x: torch.Tensor,
        mean_p: torch.Tensor,
        std_p: torch.Tensor,
        mean_q: torch.Tensor,
        std_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of mixture distribution M = 0.5 * P + 0.5 * Q.
        
        From paper: m(x) = 0.5 * p(x) + 0.5 * q(x)
        log m(x) = log(0.5 * p(x) + 0.5 * q(x))
        
        Using log-sum-exp trick for numerical stability:
        log(0.5*p + 0.5*q) = log(0.5) + log(p + q)
                           = log(0.5) + log_sum_exp([log p, log q])
        
        Args:
            x: Sample points, shape (batch_size, n_samples, output_dim)
            mean_p, std_p: Parameters of distribution P (GP)
            mean_q, std_q: Parameters of distribution Q (NN)
            
        Returns:
            Log mixture probabilities, shape (batch_size, n_samples)
        """
        # Compute log probabilities for both distributions
        log_p = self.gaussian_log_prob(x, mean_p, std_p)  # (batch, n_samples)
        log_q = self.gaussian_log_prob(x, mean_q, std_q)  # (batch, n_samples)
        
        # Stack for log-sum-exp: (batch, n_samples, 2)
        log_probs = torch.stack([log_p, log_q], dim=-1)
        
        # log m(x) = log(0.5) + log_sum_exp([log p, log q])
        # log(0.5) = -log(2)
        log_mixture = -np.log(2) + torch.logsumexp(log_probs, dim=-1)
        
        return log_mixture
    
    def monte_carlo_kl(
        self,
        samples: torch.Tensor,
        log_p_samples: torch.Tensor,
        log_m_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Monte Carlo approximation of KL divergence.
        
        From paper (Eq. 7):
        KL(P || M) ≈ (1/L) * Σ_{l=1}^{L} [log p(x^(l)) - log m(x^(l))]
        
        Args:
            samples: Samples from distribution P, shape (batch_size, n_samples, output_dim)
            log_p_samples: log p(x) for each sample, shape (batch_size, n_samples)
            log_m_samples: log m(x) for each sample, shape (batch_size, n_samples)
            
        Returns:
            KL divergence estimate, shape (batch_size,)
        """
        # KL(P||M) ≈ (1/L) * Σ [log p(x) - log m(x)]
        kl = torch.mean(log_p_samples - log_m_samples, dim=-1)
        
        # Clamp to avoid negative values due to numerical issues
        kl = torch.clamp(kl, min=0.0)
        
        return kl
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon Divergence using Monte Carlo approximation.
        
        From paper (Eq. 5):
        JSD(P_GP || Q_NN) = 0.5 * KL(P_GP || M) + 0.5 * KL(Q_NN || M)
        
        From paper (Eq. 7), Monte Carlo approximation:
        KL(P || M) ≈ (1/L) * Σ [log p(x^(l)) - log m(x^(l))]
        where samples x^(l) are drawn from P
        
        Args:
            pred_mean: Predicted means from NN (Q), shape (batch_size, output_dim)
            pred_std: Predicted standard deviations from NN, shape (batch_size, output_dim)
            target_mean: Target means from GP (P), shape (batch_size, output_dim)
            target_std: Target standard deviations from GP, shape (batch_size, output_dim)
            
        Returns:
            JSD loss value (scalar if reduction='mean'/'sum')
        """
        batch_size, output_dim = pred_mean.shape
        device = pred_mean.device
        
        # Reshape means and stds for broadcasting: (batch_size, 1, output_dim)
        mean_p = target_mean.unsqueeze(1)  # GP distribution (P)
        std_p = target_std.unsqueeze(1)
        mean_q = pred_mean.unsqueeze(1)    # NN distribution (Q)
        std_q = pred_std.unsqueeze(1)
        
        # ============================================
        # Step 1: Compute KL(P_GP || M)
        # Draw L samples from P_GP (target/GP distribution)
        # ============================================
        # Sample from N(mean_p, std_p): x = mean + std * z, where z ~ N(0, 1)
        z_p = torch.randn(batch_size, self.n_samples, output_dim, device=device)
        samples_p = mean_p + std_p * z_p  # Shape: (batch_size, n_samples, output_dim)
        
        # Compute log p(x) for samples from P
        log_p_at_samples_p = self.gaussian_log_prob(samples_p, mean_p, std_p)
        
        # Compute log m(x) = log(0.5*p(x) + 0.5*q(x)) for samples from P
        log_m_at_samples_p = self.mixture_log_prob(samples_p, mean_p, std_p, mean_q, std_q)
        
        # Monte Carlo estimate: KL(P||M) ≈ (1/L) * Σ [log p(x) - log m(x)]
        kl_p_m = self.monte_carlo_kl(samples_p, log_p_at_samples_p, log_m_at_samples_p)
        
        # ============================================
        # Step 2: Compute KL(Q_NN || M)
        # Draw L samples from Q_NN (predicted/NN distribution)
        # ============================================
        z_q = torch.randn(batch_size, self.n_samples, output_dim, device=device)
        samples_q = mean_q + std_q * z_q  # Shape: (batch_size, n_samples, output_dim)
        
        # Compute log q(x) for samples from Q
        log_q_at_samples_q = self.gaussian_log_prob(samples_q, mean_q, std_q)
        
        # Compute log m(x) for samples from Q
        log_m_at_samples_q = self.mixture_log_prob(samples_q, mean_p, std_p, mean_q, std_q)
        
        # Monte Carlo estimate: KL(Q||M) ≈ (1/L) * Σ [log q(x) - log m(x)]
        kl_q_m = self.monte_carlo_kl(samples_q, log_q_at_samples_q, log_m_at_samples_q)
        
        # ============================================
        # Step 3: Compute JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        # ============================================
        jsd = 0.5 * kl_p_m + 0.5 * kl_q_m  # Shape: (batch_size,)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(jsd)
        elif self.reduction == 'sum':
            return torch.sum(jsd)
        else:
            return jsd


class KLDivergenceGaussian(nn.Module):
    """
    Closed-form KL Divergence between two multivariate Gaussians.
    
    From the paper (Eq. 6):
    KL(N_1 || N_2) = 0.5 * [tr(Σ_2^{-1} Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) 
                           - k + ln(|Σ_2| / |Σ_1|)]
    
    where k denotes the dimensionality of the signal.
    
    Note: This is provided for reference. For JSD with Gaussian mixture M,
    we use Monte Carlo approximation since M is not Gaussian.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        super(KLDivergenceGaussian, self).__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        mean1: torch.Tensor,
        std1: torch.Tensor,
        mean2: torch.Tensor,
        std2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL(N_1 || N_2) using closed-form for diagonal covariances.
        
        For diagonal covariances, the formula simplifies to:
        KL = 0.5 * Σ_i [log(σ_2i²/σ_1i²) + (σ_1i²/σ_2i²) + ((μ_1i - μ_2i)²/σ_2i²) - 1]
        
        Args:
            mean1, std1: Parameters of N_1 (μ_1, σ_1)
            mean2, std2: Parameters of N_2 (μ_2, σ_2)
            
        Returns:
            KL divergence, shape (batch_size,)
        """
        # Ensure numerical stability
        std1 = std1 + self.epsilon
        std2 = std2 + self.epsilon
        
        var1 = std1 ** 2
        var2 = std2 ** 2
        
        # KL divergence for diagonal Gaussians (sum over dimensions)
        # = 0.5 * Σ [log(σ_2²/σ_1²) + σ_1²/σ_2² + (μ_1-μ_2)²/σ_2² - 1]
        kl_per_dim = 0.5 * (
            torch.log(var2 / var1) +  # log(|Σ_2|/|Σ_1|) for diagonal
            var1 / var2 +              # tr(Σ_2^{-1} Σ_1) for diagonal
            ((mean1 - mean2) ** 2) / var2 -  # Mahalanobis term
            1                          # -k per dimension
        )
        
        # Sum over output dimensions
        kl = torch.sum(kl_per_dim, dim=-1)
        
        return kl


class CombinedLoss(nn.Module):
    """
    Combined loss function: JSD Loss + MSE Loss.
    
    In practice, combining JSD with MSE can improve training stability.
    MSE ensures point predictions are accurate, while JSD ensures
    distributional alignment with the GP oracle.
    """
    
    def __init__(
        self,
        jsd_weight: float = 1.0,
        mse_weight: float = 0.5,
        n_samples: int = 100,
        reduction: str = 'mean'
    ):
        """
        Initialize combined loss.
        
        Args:
            jsd_weight: Weight for JSD loss term
            mse_weight: Weight for MSE loss term
            n_samples: Number of Monte Carlo samples for JSD
            reduction: Reduction method for both losses
        """
        super(CombinedLoss, self).__init__()
        
        # Store weights for loss components
        self.jsd_weight = jsd_weight
        self.mse_weight = mse_weight
        
        # Initialize loss functions
        self.jsd_loss = JSDLoss(n_samples=n_samples, reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_mean: Predicted means from NN
            pred_std: Predicted standard deviations from NN
            target_mean: Target means from GP
            target_std: Target standard deviations from GP
            
        Returns:
            total_loss: Weighted combination of JSD and MSE
            jsd_component: JSD loss value (for logging)
            mse_component: MSE loss value (for logging)
        """
        # Compute JSD loss (distributional similarity) using Monte Carlo
        jsd = self.jsd_loss(pred_mean, pred_std, target_mean, target_std)
        
        # Compute MSE loss (point prediction accuracy)
        mse = self.mse_loss(pred_mean, target_mean)
        
        # Combine losses with weights
        total = self.jsd_weight * jsd + self.mse_weight * mse
        
        return total, jsd, mse


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.
    
    Alternative to JSD that directly maximizes the likelihood of target
    distribution under the predicted Gaussian distribution.
    
    NLL = 0.5 * log(2π * σ²) + 0.5 * ((y - μ) / σ)²
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        epsilon: float = 1e-6
    ):
        """
        Initialize Gaussian NLL Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            epsilon: Small constant for numerical stability
        """
        super(GaussianNLLLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood.
        
        Args:
            pred_mean: Predicted means, shape (batch_size, output_dim)
            pred_std: Predicted standard deviations, shape (batch_size, output_dim)
            target: Target values, shape (batch_size, output_dim)
            
        Returns:
            NLL loss value
        """
        # Ensure std is positive
        pred_std = pred_std + self.epsilon
        
        # Compute variance
        var = pred_std ** 2
        
        # Compute NLL = 0.5 * log(2π * σ²) + 0.5 * ((y - μ) / σ)²
        # First term: log normalization constant
        log_term = 0.5 * torch.log(2 * np.pi * var)
        
        # Second term: squared error normalized by variance
        sq_error_term = 0.5 * ((target - pred_mean) ** 2) / var
        
        # Total NLL
        nll = log_term + sq_error_term
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            return nll


def create_loss_function(
    loss_type: str = 'jsd',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('jsd', 'combined', 'nll', or 'mse')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Initialized loss function
    """
    if loss_type == 'jsd':
        # Pure JSD loss with Monte Carlo approximation (as described in paper)
        return JSDLoss(**kwargs)
        
    elif loss_type == 'combined':
        # JSD + MSE combination (for stability)
        return CombinedLoss(**kwargs)
        
    elif loss_type == 'nll':
        # Gaussian negative log-likelihood
        return GaussianNLLLoss(**kwargs)
        
    elif loss_type == 'mse':
        # Standard MSE (baseline)
        return nn.MSELoss(**kwargs)
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    """
    Test the loss functions and verify they match the paper's formulation.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample data
    batch_size = 32
    output_dim = 9  # 9 features as in paper
    
    # Predicted distribution from neural network (Q_NN)
    pred_mean = torch.randn(batch_size, output_dim)
    pred_std = torch.abs(torch.randn(batch_size, output_dim)) + 0.1
    
    # Target distribution from GP oracle (P_GP)
    target_mean = torch.randn(batch_size, output_dim)
    target_std = torch.abs(torch.randn(batch_size, output_dim)) + 0.1
    
    print("=" * 70)
    print("Testing Loss Functions (Paper: Section 4.1)")
    print("=" * 70)
    
    # Test JSD Loss with Monte Carlo approximation
    print("\n1. Jensen-Shannon Divergence Loss (Monte Carlo, Eq. 5 & 7):")
    print("   JSD(P_GP || Q_NN) = 0.5 * KL(P_GP || M) + 0.5 * KL(Q_NN || M)")
    print("   where M = 0.5 * (P_GP + Q_NN)")
    print("   KL approximated via Monte Carlo sampling (L=100 samples)")
    
    jsd_loss = JSDLoss(n_samples=100)
    jsd_value = jsd_loss(pred_mean, pred_std, target_mean, target_std)
    print(f"   JSD Loss: {jsd_value.item():.6f}")
    
    # Verify JSD is bounded [0, log(2)]
    print(f"   Theoretical bounds: [0, {np.log(2):.6f}]")
    assert 0 <= jsd_value.item() <= np.log(2) + 0.1, "JSD out of bounds!"
    print("   ✓ JSD value within theoretical bounds")
    
    # Test closed-form KL divergence (for reference)
    print("\n2. Closed-form KL Divergence (Eq. 6, for reference):")
    print("   KL(N_1 || N_2) = 0.5 * [tr(Σ_2^{-1}Σ_1) + (μ_2-μ_1)^T Σ_2^{-1}(μ_2-μ_1) - k + ln|Σ_2|/|Σ_1|]")
    
    kl_closed = KLDivergenceGaussian()
    kl_value = kl_closed(pred_mean, pred_std, target_mean, target_std)
    print(f"   KL(Q_NN || P_GP): {kl_value.mean().item():.6f}")
    
    # Test Combined Loss
    print("\n3. Combined Loss (JSD + MSE):")
    combined_loss = CombinedLoss(jsd_weight=1.0, mse_weight=0.5, n_samples=100)
    total, jsd_comp, mse_comp = combined_loss(pred_mean, pred_std, target_mean, target_std)
    print(f"   Total Loss: {total.item():.6f}")
    print(f"   JSD Component: {jsd_comp.item():.6f}")
    print(f"   MSE Component: {mse_comp.item():.6f}")
    
    # Test Gaussian NLL Loss
    print("\n4. Gaussian Negative Log-Likelihood Loss:")
    nll_loss = GaussianNLLLoss()
    nll_value = nll_loss(pred_mean, pred_std, target_mean)
    print(f"   NLL Loss: {nll_value.item():.6f}")
    
    # Test standard MSE
    print("\n5. Standard MSE Loss (baseline):")
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(pred_mean, target_mean)
    print(f"   MSE Loss: {mse_value.item():.6f}")
    
    # Test that JSD = 0 when distributions are identical
    print("\n6. Verification: JSD should be ~0 when P = Q:")
    jsd_identical = jsd_loss(pred_mean, pred_std, pred_mean, pred_std)
    print(f"   JSD(P || P): {jsd_identical.item():.6f}")
    assert jsd_identical.item() < 0.01, "JSD should be ~0 for identical distributions!"
    print("   ✓ JSD ≈ 0 for identical distributions")
    
    print("\n" + "=" * 70)
    print("All loss functions working correctly!")
    print("Implementation matches paper's Monte Carlo JSD formulation.")
    print("=" * 70)
