#pragma once
#include "definitions.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace irspack {
namespace bpr {

using namespace std;

enum class LossType { BPR, WARP };

template <typename Real> struct BPRMFLearningConfig {
  inline BPRMFLearningConfig(size_t n_components, Real user_alpha,
                             Real item_alpha, int random_seed, bool adadelta,
                             Real learning_rate, Real rho, Real epsilon,
                             size_t n_threads)
      : n_components(n_components), user_alpha(user_alpha),
        random_seed(random_seed), adadelta(adadelta),
        learning_rate(learning_rate), rho(rho), epsilon(epsilon),
        n_threads(n_threads) {}

  BPRMFLearningConfig(const BPRMFLearningConfig<Real> &other) = default;

  const size_t n_components;
  const Real user_alpha, item_alpha;
  int random_seed;
  const bool adadelta;
  const Real learning_rate, rho, epsilon;
  size_t n_threads;

  struct Builder {
    size_t n_components = 16;
    Real user_alpha = 0;
    Real item_alpha = 0;
    int random_seed = 42;
    const bool adadelta = true;
    Real learning_rate = 1e-5;
    Real rho = .95;
    Real epsilon = 1e-6;
    size_t n_threads = 1;
    inline Builder() {}
    inline BPRMFLearningConfig<Real> build() {
      return BPRMFLearningConfig<Real>(n_components, user_alpha, item_alpha,
                                       random_seed, adadelta, learning_rate,
                                       rho, epsilon, n_threads);
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

    Builder &set_adadelta(bool adadelta) {
      this->adadelta = adadelta;
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
