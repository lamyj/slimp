#ifndef _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
#define _9ef486bc_b1a6_4872_b2a2_52eb0aea794c

#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "action_parameters.h"
#include "ArrayWriter.h"

/**
 * @brief Sample from a model.
 * @param name Base name of the model, this function will sample from
 *             <name>_sampler
 * @param data Dictionary of data passed to the sampler
 * @param parameters Sampling parameters
 * @return A dictionary containing the array of samples ("array"), the names of
 *         columns in the array ("columns") and the name of the model parameters
 *         (excluding transformed parameters and derived quantities,
 *         "parameters_columns")
 */
pybind11::dict sample(
    std::string const & name, pybind11::dict data,
    action_parameters::Sample const & parameters);

/**
 * @brief Generate quantities from a model.
 * @param name Base name of the model, this function will generate data from
 *             <name>_<variant>
 * @param variant Variant from which to generate data. This function will
 *                generate data from <name>_<variant>
 * @param data Dictionary of data
 * @param draws Array of draws from sampling
 * @param parameters Generation parameters
 * @return A dictionary containing the array of samples ("array") and the names
 *         of columns in the array ("columns") 
 */
pybind11::dict generate_quantities(
    std::string const & name, std::string const & variant,
    pybind11::dict data, Eigen::Ref<Eigen::MatrixXd> draws,
    action_parameters::GenerateQuantities const & parameters);

/**
 * @brief Compute the effective sample size for each column of the draws
 * @param draws each column hold the draws of a variable, and is concatenation
 *              of all chains
 * @param num_chains number of chains
 */
Eigen::VectorXd get_effective_sample_size(
    Eigen::Ref<Eigen::MatrixXd> draws, size_t num_chains);

/**
 * @brief Compute the potential scale reduction (Rhat) for each column of the draws
 * @param draws each column hold the draws of a variable, and is concatenation
 *              of all chains
 * @param num_chains number of chains
 */
Eigen::VectorXd get_potential_scale_reduction(
    Eigen::Ref<Eigen::MatrixXd> draws, size_t num_chains);

#endif // _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
