#pragma once
#include "definitions.hpp"
#include <cstddef>

namespace irspack {
namespace bpr {

using namespace std;

enum class LossType { BPR, WARP };
enum class SGDType { AdaGrad, AdaDelta };

template <typename Real> struct BPRMFLearningConfig {
  inline BPRMFLearningConfig(LossType loss_type, size_t n_components,
                             Real user_alpha, Real item_alpha, int random_seed,
                             SGDType sgd_type, Real learning_rate, Real rho,
                             Real epsilon, size_t n_threads)
      : loss_type(loss_type), n_components(n_components),
        user_alpha(user_alpha), item_alpha(item_alpha),
        random_seed(random_seed), sgd_type(sgd_type),
        learning_rate(learning_rate), rho(rho), epsilon(epsilon),
        n_threads(n_threads) {}

  BPRMFLearningConfig(const BPRMFLearningConfig<Real> &other) = default;

  const LossType loss_type;
  const size_t n_components;
  const Real user_alpha, item_alpha;
  const int random_seed;
  const SGDType sgd_type;
  const Real learning_rate, rho, epsilon;
  const size_t n_threads;

  struct Builder {
    LossType loss_type = LossType::BPR;
    size_t n_components = 16;
    Real user_alpha = 0;
    Real item_alpha = 0;
    int random_seed = 42;
    SGDType sgd_type = SGDType::AdaDelta;
    Real learning_rate = 1e-5;
    Real rho = .95;
    Real epsilon = 1e-6;
    size_t n_threads = 1;
    inline Builder() {}
    inline BPRMFLearningConfig<Real> build() {
      return BPRMFLearningConfig<Real>(loss_type, n_components, user_alpha,
                                       item_alpha, random_seed, sgd_type,
                                       learning_rate, rho, epsilon, n_threads);
    }

    Builder &set_loss_type(LossType loss_type) {
      this->loss_type = loss_type;
      return *this;
    }

    Builder &set_n_components(size_t n_components) {
      this->n_components = n_components;
      return *this;
    }

    Builder &set_user_alpha(Real user_alpha) {
      this->user_alpha = user_alpha;
      return *this;
    }

    Builder &set_item_alpha(Real item_alpha) {
      this->item_alpha = item_alpha;
      return *this;
    }

    Builder &set_random_seed(int random_seed) {
      this->random_seed = random_seed;
      return *this;
    }

    Builder &set_sgd_type(SGDType sgd_type) {
      this->sgd_type = sgd_type;
      return *this;
    }

    Builder &set_learning_rate(Real learning_rate) {
      this->learning_rate = learning_rate;
      return *this;
    }

    Builder &set_rho(Real rho) {
      this->rho = rho;
      return *this;
    }

    Builder &set_epsilon(Real epsilon) {
      this->epsilon = epsilon;
      return *this;
    }

    Builder &set_n_threads(size_t n_threads) {
      this->n_threads = n_threads;
      return *this;
    }
  };
};
} // namespace bpr
} // namespace irspack
