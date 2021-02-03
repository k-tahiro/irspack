#pragma once
#include "BPRMFLearningConfig.hpp"
#include "definitions.hpp"
#include <atomic>
#include <cstddef>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace irspack {
namespace bpr {

template <typename Real, typename SparseMatrix>
std::vector<std::set<int64_t>>
compute_positive_index_loc_set(const SparseMatrix &X) {
  std::vector<std::set<int64_t>> result(X.rows());
  for (int64_t u = 0; u < X.rows(); u++) {
    std::set<int64_t> &refset = result[u];
    for (typename SparseMatrix::InnerIterator it(X, u); it; ++it) {
      refset.insert(it.col());
    }
  }
  return result;
}

/*
2/3 D^2 = var
std = D * \sqrt{2/ 3}
*/

template <typename Real, class MatrixType>
void fill_random(Eigen::Ref<MatrixType> &target, Real stdev,
                 std::mt19937 &rng) {
  std::uniform_real_distribution<Real> dist(-std::sqrt(2.0 / 3.0) * stdev,
                                            std::sqrt(2.0 / 3.0) * stdev);
  for (uint64_t i = 0; i < target.rows(); i++) {
    for (uint64_t j = 0; j < target.cols(); j++) {
      target(i, j) = dist(rng);
    }
  }
}

template <typename Real> struct BPRMFTrainer {
  using ConfigType = BPRMFLearningConfig<Real>;
  using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

  using DenseMatrix =
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

  BPRMFTrainer(const ConfigType &config, const SparseMatrix &X)
      : config(config), X(X), rng(config.random_seed), n_users(X.rows()),
        n_items(X.cols()), user_factor(X.rows(), config.n_components),
        item_factor(X.cols(), config.n_components), user_bias(X.rows()),
        item_bias(X.cols()),
        positive_index_loc_set(compute_positive_index_loc_set(X)) {
    this->X.makeCompressed();
    fil_random(user_factor, static_cast<Real>(1 / 12.0) * config.n_components,
               this->rng);
    fil_random(item_factor, static_cast<Real>(1 / 12.0) * config.n_components,
               this->rng);
    user_bias.array() = static_cast<Real>(0.0);
    item_bias.array() = static_cast<Real>(0.0);

    user_factor_grad = DenseMatrix::Zero(X.rows(), config.n_components);
    user_factor_momentum = DenseMatrix::Zero(X.rows(), config.n_components);

    item_factor_grad = DenseMatrix::Zero(X.cols(), config.n_components);
    item_factor_momentum = DenseMatrix::Zero(X.cols(), config.n_components);

    user_bias_grad = DenseVector::Zero(X.rows());
    user_bias_momentum = DenseVector::Zero(X.rows());

    item_bias_grad = DenseVector::Zero(X.cols());
    item_bias_momentum = DenseVector::Zero(X.cols());

    if (config.sgd_type == SGDType::AdaGrad) {
      user_factor_grad.array() += 1.0;
      item_factor_grad.array() += 1.0;
      user_bias_grad.array() += 1.0;
      item_bias_grad.array() += 1.0;
    }
  }

  inline void step() {
    if (config.loss_type == LossType::BPR) {
      this->step_bpr();
    } else {
      throw std::runtime_error("bpr only");
    }
  };

  inline void step_bpr() { const int nnz = this->X.nonZeros(); }

  const ConfigType config;
  SparseMatrix X;
  std::mt19937 rng;
  int64_t n_users, n_items;
  DenseMatrix user_factor, item_factor;
  DenseVector user_bias, item_bias;
  std::vector<std::set<uint64_t>> positive_index_loc_set;

  DenseMatrix user_factor_grad, item_factor_grad, user_factor_momentum,
      item_factor_momentum;
  DenseVector user_bias_grad, item_bias_grad, user_bias_momentum,
      item_bias_momentum;
};
} // namespace bpr
} // namespace irspack
