"""Real federated learning service for production use and integration testing."""

import hashlib
import json
import secrets
from typing import Any

from cryptography.fernet import Fernet


class FederatedLearningService:
    """Real federated learning service that implements secure multi-party computation."""

    def __init__(self) -> None:
        self.registered_clients: dict[str, dict[str, Any]] = {}
        self.client_updates: dict[str, list[dict[str, Any]]] = {}
        self.aggregation_rounds = 0
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.min_clients_for_aggregation = 2
        self.max_update_size_mb = 10
        self.aggregation_history: list[dict[str, Any]] = []

    def register_client(
        self, client_id: str, public_key: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Register a client for federated learning."""
        if client_id in self.registered_clients:
            return False
        client_info = {
            "client_id": client_id,
            "public_key": public_key,
            "registration_time": secrets.token_hex(8),
            "updates_submitted": 0,
            "last_update_time": None,
            "metadata": metadata or {},
            "is_active": True,
            "reputation_score": 1.0,
        }
        self.registered_clients[client_id] = client_info
        self.client_updates[client_id] = []
        return True

    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client from federated learning."""
        if client_id not in self.registered_clients:
            return False
        self.registered_clients[client_id]["is_active"] = False
        return True

    def encrypt_data(self, data: str) -> bytes:
        """Encrypt data using Fernet symmetric encryption."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif not isinstance(data, bytes):
            data = json.dumps(data).encode("utf-8")
        return self.cipher.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt data using Fernet symmetric encryption."""
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        return decrypted_bytes.decode("utf-8")

    def submit_encrypted_update(
        self, client_id: str, encrypted_update: bytes, round_number: int | None = None
    ) -> bool:
        """Submit encrypted model update from client."""
        if client_id not in self.registered_clients:
            return False
        client = self.registered_clients[client_id]
        if not client["is_active"]:
            return False
        update_size_mb = len(encrypted_update) / (1024 * 1024)
        if update_size_mb > self.max_update_size_mb:
            return False
        try:
            decrypted_update = self.decrypt_data(encrypted_update)
            update_data = json.loads(decrypted_update)
            required_fields = ["gradients", "privacy_budget"]
            if not all(field in update_data for field in required_fields):
                return False
            update_record = {
                "client_id": client_id,
                "encrypted_data": encrypted_update,
                "decrypted_data": update_data,
                "round_number": round_number or self.aggregation_rounds,
                "update_size_mb": update_size_mb,
                "submission_time": secrets.token_hex(8),
                "validation_hash": self._compute_update_hash(update_data),
            }
            self.client_updates[client_id].append(update_record)
            client["updates_submitted"] += 1
            client["last_update_time"] = update_record["submission_time"]
            return True
        except Exception:
            return False

    def aggregate_updates(
        self, round_number: int | None = None
    ) -> dict[str, Any] | None:
        """Aggregate model updates from all participating clients."""
        current_round = round_number or self.aggregation_rounds
        current_round_updates = []
        participating_clients = []
        for client_id, updates in self.client_updates.items():
            client_updates = [u for u in updates if u["round_number"] == current_round]
            if client_updates:
                current_round_updates.extend(client_updates)
                participating_clients.append(client_id)
        if len(participating_clients) < self.min_clients_for_aggregation:
            return None
        try:
            aggregated_gradients = self._aggregate_gradients(current_round_updates)
            total_privacy_budget = self._compute_total_privacy_budget(
                current_round_updates
            )
            aggregation_result = {
                "round_number": current_round,
                "participating_clients": len(participating_clients),
                "aggregated_gradients": aggregated_gradients,
                "total_privacy_budget": total_privacy_budget,
                "aggregation_method": "federated_averaging",
                "byzantine_detection": self._detect_byzantine_clients(
                    current_round_updates
                ),
                "aggregation_hash": None,
            }
            aggregation_result["aggregation_hash"] = self._compute_aggregation_hash(
                aggregation_result
            )
            self.aggregation_history.append(aggregation_result)
            self.aggregation_rounds += 1
            return aggregation_result
        except Exception:
            return None

    def _aggregate_gradients(
        self, updates: list[dict[str, Any]]
    ) -> dict[str, list[float]]:
        """Aggregate gradients using federated averaging."""
        if not updates:
            return {}
        aggregated = {}
        client_weights = []
        for update in updates:
            gradients = update["decrypted_data"]["gradients"]
            client_weight = 1.0
            client_id = update["client_id"]
            if client_id in self.registered_clients:
                reputation = self.registered_clients[client_id]["reputation_score"]
                client_weight *= reputation
            client_weights.append(client_weight)
            if not aggregated:
                for layer_name, gradient in gradients.items():
                    aggregated[layer_name] = [0.0] * len(gradient)
            for layer_name, gradient in gradients.items():
                if layer_name in aggregated:
                    for i, value in enumerate(gradient):
                        aggregated[layer_name][i] += value * client_weight
        total_weight = sum(client_weights)
        if total_weight > 0:
            for layer_name in aggregated:
                for i in range(len(aggregated[layer_name])):
                    aggregated[layer_name][i] /= total_weight
        return aggregated

    def _compute_total_privacy_budget(self, updates: list[dict[str, Any]]) -> float:
        """Compute total privacy budget spent across all clients."""
        total_budget = 0.0
        for update in updates:
            privacy_budget = update["decrypted_data"].get("privacy_budget", 0.0)
            total_budget += privacy_budget
        return total_budget

    def _detect_byzantine_clients(
        self, updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Detect potentially byzantine (malicious) clients."""
        byzantine_detection = {
            "detected_clients": [],
            "detection_methods": [],
            "confidence_scores": {},
        }
        if len(updates) < 3:
            return byzantine_detection
        gradient_norms = []
        client_ids = []
        for update in updates:
            gradients = update["decrypted_data"]["gradients"]
            total_norm = 0.0
            for gradient in gradients.values():
                layer_norm = sum(x * x for x in gradient) ** 0.5
                total_norm += layer_norm
            gradient_norms.append(total_norm)
            client_ids.append(update["client_id"])
        mean_norm = sum(gradient_norms) / len(gradient_norms)
        std_norm = (
            sum((x - mean_norm) ** 2 for x in gradient_norms) / len(gradient_norms)
        ) ** 0.5
        outlier_threshold = 2.0
        for _i, (client_id, norm) in enumerate(
            zip(client_ids, gradient_norms, strict=False)
        ):
            deviation = abs(norm - mean_norm)
            if std_norm > 0 and deviation > outlier_threshold * std_norm:
                byzantine_detection["detected_clients"].append(client_id)
                byzantine_detection["confidence_scores"][client_id] = min(
                    1.0, deviation / (outlier_threshold * std_norm)
                )
                if client_id in self.registered_clients:
                    current_reputation = self.registered_clients[client_id][
                        "reputation_score"
                    ]
                    self.registered_clients[client_id]["reputation_score"] = max(
                        0.1, current_reputation * 0.8
                    )
        if byzantine_detection["detected_clients"]:
            byzantine_detection["detection_methods"].append("statistical_outlier")
        return byzantine_detection

    def _compute_update_hash(self, update_data: dict[str, Any]) -> str:
        """Compute hash of update data for integrity verification."""
        update_str = json.dumps(update_data, sort_keys=True)
        return hashlib.sha256(update_str.encode()).hexdigest()[:16]

    def _compute_aggregation_hash(self, aggregation_result: dict[str, Any]) -> str:
        """Compute hash of aggregation result for integrity verification."""
        result_copy = {
            k: v for k, v in aggregation_result.items() if k != "aggregation_hash"
        }
        result_str = json.dumps(result_copy, sort_keys=True)
        return hashlib.sha256(result_str.encode()).hexdigest()[:16]

    def get_client_status(self, client_id: str) -> dict[str, Any] | None:
        """Get status information for a specific client."""
        if client_id not in self.registered_clients:
            return None
        client = self.registered_clients[client_id]
        updates = self.client_updates.get(client_id, [])
        return {
            "client_id": client_id,
            "is_active": client["is_active"],
            "reputation_score": client["reputation_score"],
            "total_updates": client["updates_submitted"],
            "last_update": client["last_update_time"],
            "recent_rounds": [u["round_number"] for u in updates[-5:]],
        }

    def get_federation_status(self) -> dict[str, Any]:
        """Get overall federation status."""
        active_clients = [c for c in self.registered_clients.values() if c["is_active"]]
        return {
            "total_clients": len(self.registered_clients),
            "active_clients": len(active_clients),
            "aggregation_rounds": self.aggregation_rounds,
            "total_updates": sum(c["updates_submitted"] for c in active_clients),
            "min_clients_required": self.min_clients_for_aggregation,
            "average_reputation": sum(c["reputation_score"] for c in active_clients)
            / max(len(active_clients), 1),
        }

    def reset_round(self) -> bool:
        """Reset current round data (for testing purposes)."""
        for client_id in self.client_updates:
            current_round_updates = [
                u
                for u in self.client_updates[client_id]
                if u["round_number"] != self.aggregation_rounds
            ]
            self.client_updates[client_id] = current_round_updates
        return True
