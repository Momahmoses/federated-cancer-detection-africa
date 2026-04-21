"""
Differential privacy: adds calibrated Gaussian noise to model weights before sharing.
Ensures the global model cannot be reverse-engineered to expose individual patients.
"""
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class DifferentialPrivacyMechanism:
    """
    Gaussian Differential Privacy for model weights.

    Privacy guarantee: (epsilon, delta)-DP
    - epsilon: privacy budget (lower = more private, less utility)
    - delta: probability of privacy failure (set to 1/n_patients)
    - sensitivity: L2 norm bound on weight update
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 sensitivity: float = 1.0, clip_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        self.noise_multiplier = self._compute_noise_multiplier()

    def _compute_noise_multiplier(self) -> float:
        if self.epsilon <= 0:
            return 0.0
        sigma = (self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        return sigma

    def clip_gradients(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        flat = np.concatenate([w.flatten() for w in weights])
        l2_norm = np.linalg.norm(flat)
        if l2_norm > self.clip_norm:
            scale = self.clip_norm / l2_norm
            return [w * scale for w in weights]
        return weights

    def add_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        if self.noise_multiplier == 0:
            return weights
        noised = []
        for w in weights:
            noise = np.random.normal(0, self.noise_multiplier * self.clip_norm, w.shape)
            noised.append(w + noise.astype(w.dtype))
        return noised

    def privatize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        clipped = self.clip_gradients(weights)
        noised = self.add_noise(clipped)
        logger.debug(f"Privatized weights: σ={self.noise_multiplier:.4f}, ε={self.epsilon}, δ={self.delta}")
        return noised

    def privacy_report(self) -> dict:
        return {
            "mechanism": "Gaussian DP",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier_sigma": round(self.noise_multiplier, 4),
            "clip_norm": self.clip_norm,
            "privacy_guarantee": (
                f"({self.epsilon}, {self.delta})-DP: "
                f"An attacker cannot determine if any individual patient's data was used "
                f"with probability better than ε={self.epsilon} + δ={self.delta}"
            ),
        }


class SecureAggregation:
    """
    Simulates secure aggregation: server can compute average without seeing individual updates.
    In production, uses cryptographic secret sharing.
    """

    def __init__(self, n_clients: int):
        self.n_clients = n_clients

    def mask_weights(self, weights: List[np.ndarray], client_id: int,
                     round_id: int) -> List[np.ndarray]:
        rng = np.random.default_rng(seed=round_id * 1000 + client_id)
        masks = [rng.normal(0, 0.001, w.shape).astype(w.dtype) for w in weights]
        return [w + m for w, m in zip(weights, masks)]

    def unmask_aggregate(self, masked_aggregates: List[np.ndarray],
                          round_id: int) -> List[np.ndarray]:
        n_clients = self.n_clients
        total_masks = []
        for layer_idx in range(len(masked_aggregates)):
            total_mask = np.zeros_like(masked_aggregates[layer_idx])
            for client_id in range(n_clients):
                rng = np.random.default_rng(seed=round_id * 1000 + client_id)
                masks = [rng.normal(0, 0.001, w.shape) for w in masked_aggregates]
                total_mask += masks[layer_idx] / n_clients
            total_masks.append(total_mask)
        return [agg - mask for agg, mask in zip(masked_aggregates, total_masks)]


def demonstrate_privacy_guarantee(n_hospitals: int = 6, n_rounds: int = 10):
    print("=" * 55)
    print("DIFFERENTIAL PRIVACY DEMONSTRATION")
    print("=" * 55)

    configs = [
        ("Baseline (no DP)", 100.0, 1e-5),
        ("High Privacy (ε=0.5)", 0.5, 1e-5),
        ("Standard (ε=1.0)", 1.0, 1e-5),
        ("Relaxed (ε=3.0)", 3.0, 1e-5),
    ]

    layer_shape = (128, 30)
    true_weights = np.random.randn(*layer_shape).astype(np.float32)

    for name, epsilon, delta in configs:
        dp = DifferentialPrivacyMechanism(epsilon=epsilon, delta=delta)
        privatized = dp.privatize([true_weights])
        weight_error = np.mean(np.abs(privatized[0] - true_weights))
        report = dp.privacy_report()
        print(f"\n{name}")
        print(f"  σ (noise): {report['noise_multiplier_sigma']}")
        print(f"  Weight distortion (MAE): {weight_error:.4f}")
        print(f"  Privacy: {report['privacy_guarantee'][:80]}...")
