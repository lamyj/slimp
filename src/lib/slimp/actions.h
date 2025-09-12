#ifndef _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
#define _9ef486bc_b1a6_4872_b2a2_52eb0aea794c

#include <functional>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/pybind11.h>

#include "slimp/api.h"
#include "slimp/action_parameters.h"
#include "slimp/misc.h"
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
    pybind11::dict data, Tensor3d const & draws,
    action_parameters::Sample const & parameters);

using ResultsUpdater = std::function<void(Tensor3d const &, std::size_t)>;

/// @brief Sample different contexts from a same model in parallel.
template<typename Model>
void parallel_sample(
    slimp::VarContext const & context,
    slimp::action_parameters::Sample parameters, std::size_t R,
    std::function<void(VarContext &, std::size_t)> const & update_context,
    ResultsUpdater const & update_results);
    
/// @brief Sample different contexts from a same model in parallel.
template<typename Model>
void parallel_sample(
    slimp::VarContext const & context,
    slimp::action_parameters::Sample parameters, std::size_t R,
    std::function<bool(VarContext &, std::size_t)> const & update_context,
    ResultsUpdater const & update_results);

/// @brief Compute the effective sample size for each parameter
Tensor1d SLIMP_API get_effective_sample_size(Tensor3d const & draws);

/**
 * @brief Compute the effective sample size for each parameter. This is a
 * parallel wrapper on the outermost dimension around
 * get_effective_sample_size(draws).
 */
Tensor2d SLIMP_API get_effective_sample_size(Tensor4d const & data);

/// @brief Compute the potential scale reduction (Rhat) for each parameter
Tensor1d SLIMP_API get_potential_scale_reduction(Tensor3d const & draws);

/**
 * @brief Compute the potential scale reduction (Rhat) for each parameter. This
 * is a parallel wrapper on the outermost dimension around
 * get_potential_scale_reduction(draws).
 */
Tensor2d SLIMP_API get_potential_scale_reduction(Tensor4d const & data);

/// @brief Compute the split-chain potential scale reduction (Rhat) for each parameter
Tensor1d SLIMP_API get_split_potential_scale_reduction(Tensor3d const & draws);

/**
 * @brief Compute the split-chain potential scale reduction (Rhat) for each
 * parameter. This is a parallel wrapper on the outermost dimension around
 * get_split_potential_scale_reduction(draws).
 */
Tensor2d SLIMP_API get_split_potential_scale_reduction(Tensor4d const & data);

VarContext SLIMP_API to_context(pybind11::dict data);
}

#include "actions.txx"

#endif // _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
