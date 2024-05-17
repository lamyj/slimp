#ifndef _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
#define _9ef486bc_b1a6_4872_b2a2_52eb0aea794c

#include <string>
#include <vector>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>

#include "slimp/api.h"
#include "slimp/action_parameters.h"

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
    pybind11::dict data, Eigen::Ref<Eigen::MatrixXd> draws,
    action_parameters::GenerateQuantities const & parameters);

/**
 * @brief Compute the effective sample size for each column of the draws
 * @param draws each column hold the draws of a variable, and is concatenation
 *              of all chains
 * @param num_chains number of chains
 */
Eigen::VectorXd SLIMP_API get_effective_sample_size(
    Eigen::Ref<Eigen::MatrixXd> draws, size_t num_chains);

/**
 * @brief Compute the potential scale reduction (Rhat) for each column of the draws
 * @param draws each column hold the draws of a variable, and is concatenation
 *              of all chains
 * @param num_chains number of chains
 */
Eigen::VectorXd SLIMP_API get_potential_scale_reduction(
    Eigen::Ref<Eigen::MatrixXd> draws, size_t num_chains);

#include "actions.txx"

#endif // _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
