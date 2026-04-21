"""
Main entry point: simulate Pan-African federated cancer detection network.
6 hospitals, 20 rounds, differential privacy enabled.
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from src.federation.server import FederatedAggregator
from src.federation.client import HospitalClient
from src.privacy.differential_privacy import DifferentialPrivacyMechanism, demonstrate_privacy_guarantee

HOSPITALS = [
    "Lagos University Teaching Hospital (Nigeria)",
    "Kenyatta National Hospital (Kenya)",
    "Groote Schuur Hospital (South Africa)",
    "Korle-Bu Teaching Hospital (Ghana)",
    "Mulago National Referral Hospital (Uganda)",
    "Black Lion Hospital (Ethiopia)",
]

N_ROUNDS = 20
PRIVACY_EPSILON = 1.0
PRIVACY_DELTA = 1e-5
MIN_CLIENTS = 4


def simulate_federated_training():
    print("=" * 65)
    print("PAN-AFRICAN FEDERATED CANCER DETECTION — TRAINING SIMULATION")
    print("=" * 65)
    print(f"\nHospitals: {len(HOSPITALS)}")
    for h in HOSPITALS:
        print(f"  • {h}")
    print(f"\nConfiguration:")
    print(f"  Rounds: {N_ROUNDS} | Min clients/round: {MIN_CLIENTS}")
    print(f"  Differential Privacy: ε={PRIVACY_EPSILON}, δ={PRIVACY_DELTA}")
    print(f"  Patient data: NEVER leaves hospital servers")
    print()

    dp = DifferentialPrivacyMechanism(epsilon=PRIVACY_EPSILON, delta=PRIVACY_DELTA)
    aggregator = FederatedAggregator(min_clients=MIN_CLIENTS)
    clients = [HospitalClient(h) for h in HOSPITALS]

    global_weights = None
    round_results = []

    for round_num in range(1, N_ROUNDS + 1):
        participating = np.random.choice(
            clients,
            size=min(len(clients), MIN_CLIENTS + round_num % 2),
            replace=False,
        ).tolist()

        client_updates = []
        for client in participating:
            update = client.local_train(global_weights=global_weights, epochs=5)
            update["weights"] = dp.privatize(update["weights"])
            client_updates.append(update)

        global_weights, fed_round = aggregator.run_federation_round(client_updates, round_num)

        round_results.append({
            "round": round_num,
            "clients": len(participating),
            "loss": fed_round.aggregated_loss,
            "accuracy": fed_round.aggregated_accuracy,
        })

        if round_num % 5 == 0 or round_num == N_ROUNDS:
            logger.info(f"Round {round_num}/{N_ROUNDS} — "
                        f"Loss: {fed_round.aggregated_loss:.4f} | "
                        f"Accuracy: {fed_round.aggregated_accuracy:.4f}")

    df = pd.DataFrame(round_results)
    final_acc = df.iloc[-1]["accuracy"]
    baseline_acc = df.iloc[0]["accuracy"]

    print(f"\n{'='*65}")
    print("FEDERATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Initial accuracy:  {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
    print(f"  Final accuracy:    {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"  Improvement:       +{(final_acc - baseline_acc)*100:.1f}%")
    print(f"  Total rounds:      {N_ROUNDS}")
    print(f"  Patient data shared: 0 records (federated learning preserves privacy)")
    print()
    print("Privacy guarantee:")
    print(f"  {dp.privacy_report()['privacy_guarantee']}")
    print(f"{'='*65}")

    return df, global_weights


if __name__ == "__main__":
    df, _ = simulate_federated_training()
    print("\nTraining curve:")
    print(df[["round", "loss", "accuracy"]].to_string(index=False))
    print("\n--- Differential Privacy Demo ---")
    demonstrate_privacy_guarantee()
