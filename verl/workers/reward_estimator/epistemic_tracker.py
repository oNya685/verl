# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Epistemic Uncertainty Tracker using Neural-Linear Bandits approach.

This module implements epistemic uncertainty tracking for the reward estimator,
enabling more intelligent curriculum learning by distinguishing between prompts
that are genuinely easy versus prompts that are unfamiliar to the value estimator.

The core idea is to maintain an inverse covariance matrix P that tracks the
"coverage" of the feature space. Prompts with features orthogonal to previously
seen data will have high epistemic uncertainty.
"""

import torch
from typing import Optional, Dict, Any


class EpistemicUncertaintyTracker:
    """
    Internal component of RewardEstimatorWorkerMLPMode.
    Tracks epistemic uncertainty using Neural-Linear Bandits approach.
    
    This is NOT a separate Worker - it is instantiated and managed by
    RewardEstimatorWorkerMLPMode internally.
    
    The tracker maintains an inverse covariance matrix P ∈ R^(d×d) that tracks
    the "coverage" of the feature space. The epistemic uncertainty for a feature
    vector φ is computed as:
    
        U(x) = α_scale * sqrt(φᵀPφ / d)
    
    The inverse covariance matrix is updated using the Sherman-Morrison formula:
    
        P_t = P_{t-1} - (P_{t-1} φ φᵀ P_{t-1}) / (1 + φᵀ P_{t-1} φ)
    
    Attributes:
        feature_dim: Dimension of penultimate layer features (default: 128)
        lambda_reg: Regularization coefficient for covariance initialization
        alpha_scale: Scaling factor for uncertainty term
        device: Device for tensor operations
        dtype: Data type for computations
        inverse_covariance: The inverse covariance matrix P
        n_updates: Number of Sherman-Morrison updates performed
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        lambda_reg: float = 1.0,
        alpha_scale: float = 0.5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the EpistemicUncertaintyTracker.
        
        Args:
            feature_dim: Dimension of penultimate layer features (d=128)
            lambda_reg: Regularization coefficient for covariance initialization.
                       The inverse covariance matrix P is initialized to (1/λ)I.
            alpha_scale: Scaling factor for uncertainty term in U(x) computation.
            device: Device for tensor operations. If None, uses CPU.
            dtype: Data type for computations (default: float32 for numerical stability).
        
        Requirements: 1.1 - Initialize inverse covariance matrix P as (1/λ)I
        """
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.alpha_scale = alpha_scale
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Initialize inverse covariance matrix P = (1/λ)I
        # Shape: (feature_dim, feature_dim) = (128, 128)
        # Requirements: 1.1
        self.inverse_covariance = (1.0 / self.lambda_reg) * torch.eye(
            self.feature_dim, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Track number of updates for monitoring
        self.n_updates = 0

    def compute_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic uncertainty for a batch of features.
        Called internally by RewardEstimator.compute_estimated_reward().
        
        The uncertainty is computed as:
            U(x) = α_scale * sqrt(φᵀPφ / d)
        
        where:
            - φ is the feature vector (128-dim)
            - P is the inverse covariance matrix
            - d is the feature dimension (128)
            - α_scale is the scaling factor
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            
        Returns:
            Tensor of shape (batch_size,) containing U(x) values.
            All values are guaranteed to be non-negative.
        
        Requirements: 2.1, 2.2, 2.3
        """
        # Ensure features are on the correct device and dtype
        features = features.to(device=self.device, dtype=self.dtype)
        
        # Compute φᵀPφ for each sample in the batch
        # features: (batch_size, d)
        # inverse_covariance: (d, d)
        # P @ φᵀ: (d, batch_size)
        # φ @ P @ φᵀ diagonal: (batch_size,)
        
        # Efficient batch computation: (batch_size, d) @ (d, d) -> (batch_size, d)
        Pf = torch.matmul(features, self.inverse_covariance)  # (batch_size, d)
        
        # Compute φᵀPφ for each sample: sum of element-wise product
        # This gives us the diagonal of features @ P @ features.T
        quadratic_form = torch.sum(Pf * features, dim=1)  # (batch_size,)
        
        # Normalize by dimension and apply scaling
        # U(x) = α_scale * sqrt(φᵀPφ / d)
        normalized_uncertainty = quadratic_form / self.feature_dim
        
        # Clamp to ensure non-negativity (numerical stability)
        # Requirements: 2.3 - return non-negative scalar value
        normalized_uncertainty = torch.clamp(normalized_uncertainty, min=0.0)
        
        # Apply square root and scaling
        uncertainty = self.alpha_scale * torch.sqrt(normalized_uncertainty)
        
        return uncertainty

    def update(self, features: torch.Tensor) -> None:
        """
        Update inverse covariance matrix using Sherman-Morrison formula.
        Called internally by RewardEstimator.update_estimator().
        
        The Sherman-Morrison formula for rank-1 update:
            P_t = P_{t-1} - (P_{t-1} φ φᵀ P_{t-1}) / (1 + φᵀ P_{t-1} φ)
        
        This is applied sequentially for each feature vector in the batch.
        All computations are detached from the gradient graph to avoid memory overhead.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
        
        Requirements: 1.2, 1.3, 1.4
        """
        # Detach features from gradient graph to avoid memory overhead
        # Requirements: 1.3
        features = features.detach().to(device=self.device, dtype=self.dtype)
        
        batch_size = features.shape[0]
        
        # Process each feature vector sequentially using Sherman-Morrison
        # Requirements: 1.2
        for i in range(batch_size):
            phi = features[i]  # (d,)
            
            # Compute P @ φ
            Pphi = torch.matmul(self.inverse_covariance, phi)  # (d,)
            
            # Compute φᵀ @ P @ φ (scalar)
            phi_P_phi = torch.dot(phi, Pphi)
            
            # Compute denominator: 1 + φᵀPφ
            denominator = 1.0 + phi_P_phi
            
            # Skip update if denominator is too small (numerical stability)
            # This handles the edge case where 1 + φᵀPφ ≈ 0
            if denominator.abs() < 1e-10:
                continue
            
            # Sherman-Morrison update:
            # P_new = P - (Pφ)(Pφ)ᵀ / (1 + φᵀPφ)
            # Note: (Pφ)(Pφ)ᵀ = Pφφᵀ P
            outer_product = torch.outer(Pphi, Pphi)  # (d, d)
            self.inverse_covariance = self.inverse_covariance - outer_product / denominator
            
            self.n_updates += 1
        
        # Ensure numerical stability: matrix should remain positive semi-definite
        # Requirements: 1.4
        self._ensure_numerical_stability()

    def _ensure_numerical_stability(self) -> None:
        """
        Ensure the inverse covariance matrix remains numerically stable.
        
        This method checks for and corrects potential numerical issues:
        1. NaN or Inf values -> reset to initial state
        2. Negative eigenvalues -> clamp to small positive value
        
        Requirements: 1.4 - maintain numerical stability
        """
        # Check for NaN or Inf
        if torch.isnan(self.inverse_covariance).any() or torch.isinf(self.inverse_covariance).any():
            # Reset to initial state if numerical issues detected
            self.inverse_covariance = (1.0 / self.lambda_reg) * torch.eye(
                self.feature_dim,
                device=self.device,
                dtype=self.dtype
            )
            return
        
        # Make matrix symmetric (correct for floating point errors)
        self.inverse_covariance = 0.5 * (self.inverse_covariance + self.inverse_covariance.T)

    def get_condition_number(self) -> float:
        """
        Compute the condition number of the inverse covariance matrix.
        
        The condition number is useful for monitoring numerical stability.
        A very high condition number (e.g., > 10^6) indicates potential
        numerical issues.
        
        Returns:
            The condition number of P, or inf if computation fails.
        """
        try:
            # Use SVD to compute condition number (ratio of largest to smallest singular value)
            singular_values = torch.linalg.svdvals(self.inverse_covariance)
            if singular_values[-1] < 1e-10:
                return float('inf')
            return (singular_values[0] / singular_values[-1]).item()
        except Exception:
            return float('inf')

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state for checkpointing.
        Called by RewardEstimator when saving checkpoints.
        
        Returns:
            Dictionary containing:
                - inverse_covariance: The P matrix
                - n_updates: Number of updates performed
                - lambda_reg: Regularization coefficient
                - alpha_scale: Scaling factor
                - feature_dim: Feature dimension
        
        Requirements: 1.5
        """
        return {
            'inverse_covariance': self.inverse_covariance.cpu(),
            'n_updates': self.n_updates,
            'lambda_reg': self.lambda_reg,
            'alpha_scale': self.alpha_scale,
            'feature_dim': self.feature_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.
        Called by RewardEstimator when loading checkpoints.
        
        Args:
            state: Dictionary containing saved state from state_dict()
        
        Requirements: 1.5
        """
        self.inverse_covariance = state['inverse_covariance'].to(
            device=self.device, 
            dtype=self.dtype
        )
        self.n_updates = state['n_updates']
        
        # Optionally update hyperparameters if they were saved
        if 'lambda_reg' in state:
            self.lambda_reg = state['lambda_reg']
        if 'alpha_scale' in state:
            self.alpha_scale = state['alpha_scale']
        if 'feature_dim' in state:
            self.feature_dim = state['feature_dim']

    def to(self, device: torch.device) -> 'EpistemicUncertaintyTracker':
        """
        Move the tracker to a different device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.device = device
        self.inverse_covariance = self.inverse_covariance.to(device)
        return self

    def reset(self) -> None:
        """
        Reset the tracker to its initial state.
        
        This reinitializes the inverse covariance matrix to (1/λ)I
        and resets the update counter.
        """
        self.inverse_covariance = (1.0 / self.lambda_reg) * torch.eye(
            self.feature_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.n_updates = 0
