"""Python-authoritative SYNAPSE racing simulation."""

from .config import SimulationConfig
from .env import RacingEnv
from .evaluation import EvaluationResult, evaluate_policy, validate_evaluation_contract
from .observation import (
    ObservationSchema,
    RunningObservationNormalizer,
    TrainingContract,
    build_observation_schema,
)
from .ppo import FixedMLPActorCritic, PPOTrainer, PPOUpdateMetrics, compute_masked_gae
from .reward import RewardConfig
from .rollout import PPOHyperparameters, RolloutBatch, RolloutCollector, RolloutResult
from .track import TrackGeometry
from .vector_env import RacingVectorEnv, VectorizedRacingEnv

__all__ = [
    "ObservationSchema",
    "FixedMLPActorCritic",
    "PPOTrainer",
    "PPOUpdateMetrics",
    "RacingEnv",
    "RacingVectorEnv",
    "RewardConfig",
    "PPOHyperparameters",
    "RolloutBatch",
    "RolloutCollector",
    "RolloutResult",
    "RunningObservationNormalizer",
    "SimulationConfig",
    "TrackGeometry",
    "TrainingContract",
    "VectorizedRacingEnv",
    "build_observation_schema",
    "compute_masked_gae",
    "EvaluationResult",
    "evaluate_policy",
    "validate_evaluation_contract",
]
