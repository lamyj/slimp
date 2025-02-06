#ifndef _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
#define _9ef486bc_b1a6_4872_b2a2_52eb0aea794c

#include <functional>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/pybind11.h>
#include <xtensor/xtensor.hpp>

#include "slimp/api.h"
#include "slimp/action_parameters.h"
#include "slimp/VarContext.h"

namespace slimp
{

/**
 * @brief Sample from a model.
 * @param data Dictionary of data passed to the sampler
 * @param parameters Sampling parameters
 * @return A dictionary containing the array of samples ("array"), the names of
 *         columns in the array ("columns") and the name of the model parameters
 *         (excluding transformed parameters and derived quantities,
 *         "parameters_columns")
 */
template<typename Model>
pybind11::dict SLIMP_API sample(
    pybind11::dict data, action_parameters::Sample const & parameters);

/**
 * @brief Generate quantities from a model.
 * @param data Dictionary of data
 * @param draws Array of draws from sampling
 * @param parameters Generation parameters
 * @return A dictionary containing the array of samples ("array") and the names
 *         of columns in the array ("columns") 
 */
template<typename Model>
pybind11::dict SLIMP_API generate_quantities(
    pybind11::dict data, xt::xtensor<double, 3> const & draws,
    action_parameters::Sample const & parameters);

using ContextUpdater = std::function<void(VarContext &, std::size_t)>;
using ResultsUpdater = std::function<
    void(xt::xtensor<double, 3> const &, std::size_t)>;

/// @brief Sample different contexts from a same model in parallel.
template<typename Model>
void parallel_sample(
    slimp::VarContext const & context,
    slimp::action_parameters::Sample parameters, std::size_t R,
    ContextUpdater const & update_context,
    ResultsUpdater const & update_results);

/// @brief Compute the effective sample size for each parameter
xt::xtensor<double, 1> SLIMP_API get_effective_sample_size(
    xt::xtensor<double, 3> const & draws);

/// @brief Compute the potential scale reduction (Rhat) for each parameter
xt::xtensor<double, 1> SLIMP_API get_potential_scale_reduction(
    xt::xtensor<double, 3> const & draws);

/// @brief Compute the split-chain potential scale reduction (Rhat) for each parameter
xt::xtensor<double, 1> SLIMP_API get_split_potential_scale_reduction(
    xt::xtensor<double, 3> const & draws);

}

#include "actions.txx"

#endif // _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
