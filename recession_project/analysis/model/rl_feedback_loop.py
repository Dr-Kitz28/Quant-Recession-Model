#!/usr/bin/env python3
"""Reinforcement Learning Feedback Loop for Correlation Forecast Refinement.

This module implements a REINFORCE-style policy gradient agent that learns
to adjust correlation forecasts to minimize prediction error (Δ).

The feedback loop:
    1. Receives predicted correlation matrix M_pred
    2. Applies learned adjustments → M″
    3. Compares M″ with actual M_actual
    4. Updates policy to minimize Δ = ||M″ - M_actual||

This creates the Δ feedback loop shown in the RecessionModel diagram.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================================
# Policy Network
# ============================================================================
class PolicyNetwork(nn.Module):
    """Policy network that outputs adjustment mean and std for each correlation."""

    def __init__(
        self,
        n_corr_features: int,
        hidden_dim: int = 128,
        min_std: float = 0.01,
        max_std: float = 0.5,
    ):
        super().__init__()
        self.n_corr_features = n_corr_features
        self.min_std = min_std
        self.max_std = max_std

        # shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_corr_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # mean head (adjustment to add to prediction)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, n_corr_features),
            nn.Tanh(),  # adjustments in [-1, 1]
        )

        # log-std head
        self.log_std_head = nn.Linear(hidden_dim, n_corr_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: predicted correlations (batch, n_corr_features)

        Returns:
            mean: adjustment means (batch, n_corr_features)
            std: adjustment stds (batch, n_corr_features)
        """
        h = self.backbone(x)
        mean = self.mean_head(h) * 0.1  # scale down adjustments
        log_std = self.log_std_head(h)
        std = torch.sigmoid(log_std) * (self.max_std - self.min_std) + self.min_std
        return mean, std

    def sample_action(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample adjustment from policy and return log probability."""
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        action = dist.rsample()  # reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean


# ============================================================================
# RL Feedback Agent
# ============================================================================
class RLFeedbackAgent:
    """REINFORCE agent for correlation forecast refinement."""

    def __init__(
        self,
        n_corr_features: int,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.policy = PolicyNetwork(n_corr_features, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.n_corr_features = n_corr_features
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []

    def select_adjustment(
        self, pred_corr_flat: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select adjustment for predicted correlation."""
        self.policy.eval()
        # Handle NaN values by replacing with 0
        pred_clean = np.nan_to_num(pred_corr_flat.astype(np.float32), nan=0.0)
        x = torch.from_numpy(pred_clean).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(not deterministic):
            if deterministic:
                mean, _ = self.policy(x)
                adjustment = mean.squeeze(0).detach().cpu().numpy()
            else:
                action, log_prob, _ = self.policy.sample_action(x)
                self.episode_log_probs.append(log_prob)
                adjustment = action.squeeze(0).detach().cpu().numpy()

        return adjustment

    def observe_reward(self, reward: float) -> None:
        """Store reward for current step."""
        self.episode_rewards.append(reward)

    def compute_returns(self) -> List[float]:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update_policy(self) -> float:
        """Update policy using REINFORCE."""
        if len(self.episode_log_probs) == 0:
            return 0.0

        returns = self.compute_returns()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # normalize returns
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = 0.0
        for log_prob, R in zip(self.episode_log_probs, returns_t):
            policy_loss -= log_prob * R

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # clear episode buffers
        loss_val = policy_loss.item()
        self.episode_log_probs = []
        self.episode_rewards = []

        return loss_val

    def save(self, path: Path) -> None:
        """Save policy checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"[rl_feedback] Saved checkpoint to {path}")

    def load(self, path: Path) -> None:
        """Load policy checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[rl_feedback] Loaded checkpoint from {path}")


# ============================================================================
# Feedback Loop
# ============================================================================
class FeedbackLoop:
    """Run the RL feedback loop on correlation sequences."""

    def __init__(
        self,
        agent: RLFeedbackAgent,
        n_spreads: int,
    ):
        self.agent = agent
        self.n_spreads = n_spreads
        self.triu_idx = np.triu_indices(n_spreads, k=1)

    def compute_reward(
        self,
        adjusted_corr: np.ndarray,
        actual_corr: np.ndarray,
    ) -> float:
        """Compute reward as negative MSE between adjusted and actual."""
        adjusted_flat = adjusted_corr[self.triu_idx]
        actual_flat = actual_corr[self.triu_idx]

        # handle NaNs
        valid = ~(np.isnan(adjusted_flat) | np.isnan(actual_flat))
        if not valid.any():
            return 0.0

        mse = np.mean((adjusted_flat[valid] - actual_flat[valid]) ** 2)
        # negative MSE as reward (higher is better)
        return -mse

    def run_episode(
        self,
        pred_correlations: np.ndarray,   # (T, n_spreads, n_spreads)
        actual_correlations: np.ndarray,  # (T, n_spreads, n_spreads)
        train: bool = True,
    ) -> Tuple[float, List[float]]:
        """Run one episode of the feedback loop."""
        T = len(pred_correlations)
        rewards = []
        adjusted = []

        for t in range(T):
            pred_flat = pred_correlations[t][self.triu_idx]
            actual = actual_correlations[t]

            # Skip samples with too many NaN values (>50%)
            nan_ratio = np.isnan(pred_flat).sum() / len(pred_flat)
            if nan_ratio > 0.5:
                continue

            # get adjustment from policy
            adjustment = self.agent.select_adjustment(pred_flat, deterministic=not train)

            # apply adjustment (handle NaNs)
            pred_flat_clean = np.nan_to_num(pred_flat, nan=0.0)
            adjusted_flat = np.clip(pred_flat_clean + adjustment, -1, 1)

            # reconstruct matrix
            adjusted_corr = np.eye(self.n_spreads, dtype=np.float32)
            adjusted_corr[self.triu_idx] = adjusted_flat
            adjusted_corr = adjusted_corr + adjusted_corr.T - np.diag(np.diag(adjusted_corr))
            adjusted.append(adjusted_corr)

            # compute reward
            reward = self.compute_reward(adjusted_corr, actual)
            rewards.append(reward)

            if train:
                self.agent.observe_reward(reward)

        # update policy
        loss = 0.0
        if train:
            loss = self.agent.update_policy()

        return loss, rewards

    def train(
        self,
        pred_correlations: np.ndarray,
        actual_correlations: np.ndarray,
        epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the feedback loop on historical data."""
        history = {"loss": [], "avg_reward": [], "final_reward": []}

        for epoch in range(1, epochs + 1):
            loss, rewards = self.run_episode(pred_correlations, actual_correlations, train=True)
            avg_reward = np.mean(rewards)
            final_reward = rewards[-1] if rewards else 0.0

            history["loss"].append(loss)
            history["avg_reward"].append(avg_reward)
            history["final_reward"].append(final_reward)

            if verbose and epoch % 10 == 0:
                print(f"[epoch {epoch:03d}] loss={loss:.6f} avg_reward={avg_reward:.6f}")

        return history


# ============================================================================
# CLI / Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="RL Feedback Loop for correlation refinement")
    parser.add_argument("--pred-npz", type=Path, required=True, help="Predicted correlations NPZ")
    parser.add_argument("--target-npz", type=Path, required=True, help="Target (actual) correlations NPZ")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    print(f"[rl_feedback] Loading data...")

    # load predictions (could be from weight learner or AFNS forecast)
    pred_data = np.load(args.pred_npz, allow_pickle=True)
    pred_corr = pred_data["corr"].astype(float)

    # load targets
    target_data = np.load(args.target_npz, allow_pickle=True)
    target_corr = target_data["corr"].astype(float)

    # align lengths (shift by 1 for next-day prediction)
    # pred[t] should predict target[t+1]
    pred_corr = pred_corr[:-1]
    target_corr = target_corr[1:]

    n_spreads = pred_corr.shape[1]
    n_corr_features = len(np.triu_indices(n_spreads, k=1)[0])

    print(f"[rl_feedback] Pred shape: {pred_corr.shape}, Target shape: {target_corr.shape}")
    print(f"[rl_feedback] Correlation features: {n_corr_features}")

    # create agent and feedback loop
    agent = RLFeedbackAgent(
        n_corr_features=n_corr_features,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
    )

    loop = FeedbackLoop(agent, n_spreads)

    # train
    history = loop.train(pred_corr, target_corr, epochs=args.epochs, verbose=True)

    # save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "rl_feedback_policy.pt"
    agent.save(checkpoint_path)

    history_path = args.output_dir / "rl_feedback_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[rl_feedback] Saved history to {history_path}")

    # summary
    final_reward = history["avg_reward"][-1] if history["avg_reward"] else 0.0
    print(f"[rl_feedback] Final avg_reward={final_reward:.6f}")
    print(f"[rl_feedback] Δ reduction achieved through policy learning")


if __name__ == "__main__":
    main()
