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

#include "ArrayWriter.h"

/**
 * @brief Sample from a model.
 * @param name Base name of the model, this function will sample from
 *             <name>_sampler
 * @param data Dictionary of data passed to the sampler
 * @return Array of samples and names of columns
 */
std::tuple<ArrayWriter::Array, std::vector<std::string>> sample(
    std::string const & name, pybind11::dict data);

/**
 * @brief Generate quantities from a model.
 * @param name Base name of the model, this function will generate data from
 *             <name>_<variant>
 * @param variant Variant from which to generate data. This function will
 *                generate data from <name>_<variant>
 * @param data Dictionary of data
 * @param draws Array of draws from sampling
 * @return Array of generated quantities and names of columns
 */
std::tuple<ArrayWriter::Array, std::vector<std::string>> generate_quantities(
    std::string const & name, std::string const & variant,
    pybind11::dict data, Eigen::Ref<Eigen::MatrixXd> draws);

#endif // _9ef486bc_b1a6_4872_b2a2_52eb0aea794c
