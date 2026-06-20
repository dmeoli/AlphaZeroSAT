#include "minisat/core/Const.h"
#include <random>

const float Hyper_Const::c_act = 0.05854f;    // need a better value here for exploration
const int Hyper_Const::MCTS_size_lim = 100; // the size of MCT we want to achieve.

//const double Hyper_Const::alpha[Hyper_Const::nact] = {0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3};

const double Hyper_Const::alpha[Hyper_Const::nact] = {2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};

void Hyper_Const::generate_dirichlet(double* di) {
    // Dirichlet(alpha) drawn as normalised Gamma(alpha_i, 1) samples; this is exactly
    // what gsl_ran_dirichlet did, but with the C++11 <random> facilities (no GSL).
    static std::mt19937 gen(5489u);  // mt19937, matching the original gsl_rng_mt19937 choice
    double sum = 0.0;
    for (int i = 0; i < Hyper_Const::nact; ++i) {
        std::gamma_distribution<double> gamma(Hyper_Const::alpha[i], 1.0);
        di[i] = gamma(gen);
        sum += di[i];
    }
    if (sum > 0.0)
        for (int i = 0; i < Hyper_Const::nact; ++i) di[i] /= sum;
}
//const float Hyper_Const::c_act = 6.095f;      // need a better value here for exploration
//const int Hyper_Const::MCTS_size_lim = 487; // the size of MCT we want to achieve.
