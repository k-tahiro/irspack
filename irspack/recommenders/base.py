from abc import ABC, abstractmethod
from os import environ
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from optuna.trial import Trial
from scipy import sparse as sps

if TYPE_CHECKING:
    from .. import evaluator

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)


class CallBeforeFitError(Exception):
    pass


class BaseRecommender(ABC):
    """The base class for all (hot) recommenders.

    Args:
        X_train_all (csr_matrix|csc_matrix|np.ndarray): user/item interaction matrix.
            each row correspods to a user's interaction with items.
    """

    def __init__(self, X_train_all: InteractionMatrix, **kwargs: Any) -> None:

        self.X_train_all = sps.csr_matrix(X_train_all).astype(np.float64)
        self.n_users: int = self.X_train_all.shape[0]
        self.n_items: int = self.X_train_all.shape[1]
        self.X_train_all.sort_indices()

        # this will store configurable parameters learnt during the training,
        # e.g., the epoch with the best validation score.
        self.learnt_config: Dict[str, Any] = dict()

    def learn(self) -> "BaseRecommender":
        """Learns and returns itself.

        Returns:
            BaseRecommender: The model after fitting process.
        """
        self._learn()
        return self

    @abstractmethod
    def _learn(self) -> None:
        pass

    def learn_with_optimizer(
        self, evaluator: Optional["evaluator.Evaluator"], trial: Optional[Trial]
    ) -> None:
        """Learning procedures with early stopping and pruning.

        Args:
            evaluator (Optional[): The evaluator to measure the score.
            trial (Optional[Trial]): The current optuna trial under the study (if any.)
        """
        # by default, evaluator & trial does not play any role.
        self.learn()

    @abstractmethod
    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        """Compute the item recommendation score for a subset of users.

        Args:
            user_indices (UserIndexArray): The index defines the subset of users.

        Returns:
            DenseScoreArray: The item scores. Its shape will be (len(user_indices), self.n_items)
        """
        raise NotImplementedError("get_score must be implemented")  # pragma: no cover

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        """Compute the score for a block of the users.

        Args:
            begin (int): where the evaluated user block begins.
            end (int): where the evaluated user block ends.

        Returns:
            DenseScoreArray: The item scores. Its shape will be (end - begin, self.n_items)
        """
        raise NotImplementedError(
            "get_score_block not implemented!"
        )  # pragma: no cover

    def get_score_remove_seen(self, user_indices: UserIndexArray) -> DenseScoreArray:
        """Compute the item score and mask the item in the training set.
            Masked items will have the score -inf.

        Args:
            user_indices (UserIndexArray): Specifies the subset of users.

        Returns:
            DenseScoreArray: The masked item scores. Its shape will be (len(user_indices), self.n_items)
        """
        scores = self.get_score(user_indices)
        if sps.issparse(scores):
            scores = scores.toarray()
        m = self.X_train_all[user_indices].tocsr()
        scores[m.nonzero()] = -np.inf
        if scores.dtype != np.float64:
            scores = scores.astype(np.float64)
        return scores

    def get_score_remove_seen_block(self, begin: int, end: int) -> DenseScoreArray:
        """Compute the score for a block of the users, and mask the items in the training set.
            Masked items will have the score -inf.

        Args:
            begin (int): where the evaluated user block begins.
            end (int): where the evaluated user block ends.

        Returns:
            DenseScoreArray: The masked item scores. Its shape will be (end - begin, self.n_items)
        """
        scores = self.get_score_block(begin, end)
        if sps.issparse(scores):
            scores = scores.toarray()
        m = self.X_train_all[begin:end]
        scores[m.nonzero()] = -np.inf
        if scores.dtype != np.float64:
            scores = scores.astype(np.float64)
        return scores

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        """Compute the item recommendation score for unseen users whose profiles are given as another user-item relation matrix.

        Args:
            X (InteractionMatrix): The profile user-item relation matrix for unseen users.
                Its number of rows is arbitrary, but the number of columns must be self.n_items.

        Returns:
            DenseScoreArray: Computed item scores for users. Its shape is equal to X.
        """
        raise NotImplementedError(
            f"get_score_cold_user is not implemented for {self.__class__.__name__}!"
        )  # pragma: no cover

    def get_score_cold_user_remove_seen(self, X: InteractionMatrix) -> DenseScoreArray:
        """Compute the item recommendation score for unseen users whose profiles are given as another user-item relation matrix.
            The score will then be masked by the input.

        Args:
            X (InteractionMatrix): The profile user-item relation matrix for unseen users.
                Its number of rows is arbitrary, but the number of columns must be self.n_items.

        Returns:
            DenseScoreArray: Computed & masked item scores for users. Its shape is equal to X.
        """
        score = self.get_score_cold_user(X)
        score[X.nonzero()] = -np.inf
        return score


class BaseRecommenderWithThreadingSupport(BaseRecommender):
    def __init__(
        self, X_train_all: InteractionMatrix, n_thread: Optional[int], **kwargs: Any
    ):

        super(BaseRecommenderWithThreadingSupport, self).__init__(X_train_all, **kwargs)
        if n_thread is not None:
            self.n_thread = n_thread
        else:
            try:
                self.n_thread = int(environ.get("IRSPACK_NUM_THREADS_DEFAULT", "1"))
            except:
                raise ValueError(
                    'failed to interpret "IRSPACK_NUM_THREADS_DEFAULT" as an integer.'
                )


class BaseSimilarityRecommender(BaseRecommender):
    W_: Optional[Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(BaseSimilarityRecommender, self).__init__(*args, **kwargs)
        self.W_ = None

    @property
    def W(self) -> Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]:
        if self.W_ is None:
            raise RuntimeError("W fetched before fit.")
        return self.W_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        if sps.issparse(self.W):
            return self.X_train_all[user_indices].dot(self.W).toarray()
        else:
            return self.X_train_all[user_indices].dot(self.W)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        if self.W is None:
            raise RuntimeError("'get_score_cold_user' called before the fit")
        if sps.issparse(self.W):
            return X.dot(self.W).toarray()
        else:
            return X.dot(self.W)

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        if sps.issparse(self.W):
            return self.X_train_all[begin:end].dot(self.W).toarray()
        else:
            return self.X_train_all[begin:end].dot(self.W)


class BaseRecommenderWithUserEmbedding(BaseRecommender):
    """Defines a recommender with user embedding (e.g., matrix factorization.).
    These class can be a base CF estimator for CB2CF (with user profile -> user embedding NN).
    """

    @abstractmethod
    def get_user_embedding(
        self,
    ) -> DenseMatrix:
        pass

    @abstractmethod
    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        pass


class BaseRecommenderWithItemEmbedding(BaseRecommender):
    """Defines a recommender with item embedding (e.g., matrix factorization.).
    These class can be a base CF estimator for CB2CF (with item profile -> item embedding NN).
    """

    @abstractmethod
    def get_item_embedding(
        self,
    ) -> DenseMatrix:
        raise NotImplementedError(
            "get_item_embedding must be implemented"
        )  # pragma: no cover

    @abstractmethod
    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        pass  # pragma: no cover
