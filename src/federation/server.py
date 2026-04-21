"""
Federated learning aggregation server using Flower (flwr).
Hospitals train locally, only model weights (never patient data) are shared here.
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    from flwr.common import Metrics, Parameters, Scalar
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    logger.warning("Flower (flwr) not installed. Using mock federation.")


@dataclass
class FedRound:
    round_number: int
    n_clients: int
    aggregated_loss: float
    aggregated_accuracy: float
    participating_hospitals: List[str] = field(default_factory=list)


class FederatedAggregator:
    """
    Federated Averaging (FedAvg) aggregation.
    Weighted by number of local training samples per hospital.
    """

    def __init__(self, min_clients: int = 3, fraction_fit: float = 1.0):
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.round_history: List[FedRound] = []
        self.global_weights: Optional[List[np.ndarray]] = None

    def federated_average(self, client_weights: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        FedAvg: weighted average of client weights.
        client_weights: list of (weights, num_samples) tuples.
        """
        total_samples = sum(n for _, n in client_weights)
        averaged = []
        for layer_idx in range(len(client_weights[0][0])):
            weighted_sum = np.zeros_like(client_weights[0][0][layer_idx], dtype=np.float64)
            for weights, n_samples in client_weights:
                weighted_sum += weights[layer_idx].astype(np.float64) * (n_samples / total_samples)
            averaged.append(weighted_sum.astype(np.float32))
        return averaged

    def run_federation_round(
        self,
        client_updates: List[Dict],
        round_number: int,
    ) -> Tuple[List[np.ndarray], FedRound]:
        logger.info(f"Round {round_number}: Aggregating {len(client_updates)} hospital updates")

        weight_samples = [
            (update["weights"], update["n_samples"])
            for update in client_updates
        ]
        aggregated_weights = self.federated_average(weight_samples)
        self.global_weights = aggregated_weights

        avg_loss = np.mean([u.get("loss", 0) for u in client_updates])
        avg_acc = np.mean([u.get("accuracy", 0) for u in client_updates])

        fed_round = FedRound(
            round_number=round_number,
            n_clients=len(client_updates),
            aggregated_loss=round(float(avg_loss), 4),
            aggregated_accuracy=round(float(avg_acc), 4),
            participating_hospitals=[u.get("hospital_id", f"H{i}") for i, u in enumerate(client_updates)],
        )
        self.round_history.append(fed_round)
        logger.info(f"Round {round_number} complete: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
        return aggregated_weights, fed_round

    def build_flower_strategy(self):
        if not FLWR_AVAILABLE:
            return None

        def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
            total = sum(n for n, _ in metrics)
            acc = sum(n * m.get("accuracy", 0) for n, m in metrics) / total
            loss = sum(n * m.get("loss", 0) for n, m in metrics) / total
            return {"accuracy": acc, "loss": loss}

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average,
        )
        return strategy

    def start_server(self, host: str = "0.0.0.0", port: int = 8080, n_rounds: int = 20):
        if not FLWR_AVAILABLE:
            logger.info("Running mock federation (Flower not installed)")
            return self._mock_federation(n_rounds)

        strategy = self.build_flower_strategy()
        fl.server.start_server(
            server_address=f"{host}:{port}",
            config=fl.server.ServerConfig(num_rounds=n_rounds),
            strategy=strategy,
        )

    def _mock_federation(self, n_rounds: int = 5) -> List[FedRound]:
        logger.info("Starting mock federated learning simulation...")
        hospitals = ["Lagos University Teaching Hospital", "Kenyatta National Hospital",
                     "Groote Schuur Hospital", "Teaching Hospital Accra",
                     "Mulago National Referral Hospital", "Black Lion Hospital Addis Ababa"]

        model_size = [(64, 32), (32,), (32, 16), (16,), (16, 2), (2,)]
        weights = [np.random.randn(*shape).astype(np.float32) for shape in model_size]

        for round_num in range(1, n_rounds + 1):
            client_updates = []
            for hospital in hospitals[:self.min_clients + round_num % 3]:
                n_samples = np.random.randint(200, 2000)
                noise = 0.1 / round_num
                local_weights = [w + np.random.randn(*w.shape) * noise for w in weights]
                client_updates.append({
                    "hospital_id": hospital,
                    "weights": local_weights,
                    "n_samples": n_samples,
                    "loss": max(0.05, 0.8 - round_num * 0.05 + np.random.uniform(-0.05, 0.05)),
                    "accuracy": min(0.97, 0.55 + round_num * 0.04 + np.random.uniform(-0.02, 0.02)),
                })

            weights, fed_round = self.run_federation_round(client_updates, round_num)

        return self.round_history
