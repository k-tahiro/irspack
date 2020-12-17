from sklearn.decomposition import NMF
from .base import BaseRecommender
from ..definitions import InteractionMatrix, DenseScoreArray, UserIndexArray
from .. import parameter_tuning


class NMFRecommender(BaseRecommender):
    default_tune_range = [
        parameter_tuning.IntegerSuggestion("n_components", 4, 512),
        parameter_tuning.LogUniformSuggestion("alpha", 1e-10, 1e-1),
        parameter_tuning.UniformSuggestion("l1_ratio", 0, 1),
        parameter_tuning.CategoricalSuggestion(
            "beta_loss", ["frobenius", "kullback-leibler"]
        ),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        n_components: int = 64,
        alpha: float = 1e-2,
        l1_ratio: float = 1e-2,
        beta_loss: str = "frobenius",
        init=None,
    ):
        super().__init__(X_all)
        self.n_components = n_components
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.beta_loss = beta_loss
        self.init = init

    def _learn(self) -> None:
        nmf_model = NMF(
            n_components=self.n_components,
            alpha=self.alpha,
            init=self.init,
            l1_ratio=self.l1_ratio,
            beta_loss=self.beta_loss,
            random_state=42,
        )
        self.nmf_model = nmf_model
        self.nmf_model.fit(self.X_all)
        self.W = self.nmf_model.fit_transform(self.X_all.tocsr())
        self.H = self.nmf_model.components_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return self.W[user_indices].dot(self.H)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        return self.nmf_model.transform(X).dot(self.H)
