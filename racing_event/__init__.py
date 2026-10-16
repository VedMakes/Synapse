"""Python-authoritative SYNAPSE racing simulation."""

import importlib.util

from .config import SimulationConfig
from .env import RacingEnv
from .evaluation import EvaluationResult, evaluate_policy, validate_evaluation_contract
from .observation import (
    ObservationSchema,
    RunningObservationNormalizer,
    TrainingContract,
    build_observation_schema,
)
from .reward import RewardConfig, RewardFunction, evaluate_reward_function
from .rollout import PPOHyperparameters, RolloutBatch, RolloutCollector, RolloutResult
from .track import TrackGeometry
from .vector_env import RacingVectorEnv, VectorizedRacingEnv

_PPO_EXPORTS = {"FixedMLPActorCritic", "PPOTrainer", "PPOUpdateMetrics", "compute_masked_gae"}

__all__ = [
    "ObservationSchema",
    "RacingEnv",
    "RacingVectorEnv",
    "RewardConfig",
    "RewardFunction",
    "evaluate_reward_function",
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
    "EvaluationResult",
    "evaluate_policy",
    "validate_evaluation_contract",
]

if importlib.util.find_spec("torch") is not None:
    __all__.extend(sorted(_PPO_EXPORTS))


def __getattr__(name: str):
    """Load the optional PyTorch reference trainer only when requested."""

    if name in _PPO_EXPORTS:
        from . import ppo

        return getattr(ppo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
