from typing import List, Any
from abc import ABC, abstractmethod

from ..recommenders.base import InteractionMatrix
from ..definitions import DenseScoreArray, ProfileMatrix
from ..parameter_tuning import Suggestion


class BaseUserColdStartRecommender(ABC):
    default_tune_range: List[Suggestion] = []

    def __init__(
        self,
        X_interaction: InteractionMatrix,
        X_profile: ProfileMatrix,
        **kwargs: Any
    ):
        assert X_interaction.shape[0] == X_profile.shape[0]
        self.n_user = X_profile.shape[0]
        self.n_item = X_interaction.shape[1]
        self.profile_dimension = X_profile.shape[1]
        self.X_profile = X_profile
        self.X_interaction = X_interaction

    @abstractmethod
    def _learn(self) -> None:
        pass

    def learn(self) -> "BaseUserColdStartRecommender":
        self._learn()
        return self

    @abstractmethod
    def get_score(self, profile: ProfileMatrix) -> DenseScoreArray:
        raise NotImplementedError("implemented in the descendant")
