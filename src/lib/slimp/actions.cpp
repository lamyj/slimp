#include "actions.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stan/callbacks/interrupt.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/standalone_gqs.hpp>

#include "slimp/action_parameters.h"
#include "slimp/ArrayWriter.h"
#include "slimp/Factory.h"
#include "slimp/Logger.h"
#include "slimp/VarContext.h"

Eigen::VectorXd get_effective_sample_size(
    Eigen::Ref<Eigen::MatrixXd> draws, size_t num_chains)
{
    Eigen::VectorXd sample_size(draws.cols());
    
    auto const draws_per_chain = draws.rows()/num_chains;
    for(size_t column=0; column!=draws.cols(); ++column)
    {
        auto const vector = draws.col(column);
        
        std::vector<double const *> chains(num_chains);
        for(size_t chain=0; chain!=num_chains; ++chain)
        {
            chains[chain] = vector.data()+draws_per_chain*chain;
        }
        sample_size[column] = stan::analyze::compute_effective_sample_size(
            chains, draws_per_chain);
    }
    
    return sample_size;
}

Eigen::VectorXd get_potential_scale_reduction(
    Eigen::Ref<Eigen::MatrixXd> draws, size_t num_chains)
{
    Eigen::VectorXd sample_size(draws.cols());
    
    auto const draws_per_chain = draws.rows()/num_chains;
    for(size_t column=0; column!=draws.cols(); ++column)
    {
        auto const vector = draws.col(column);
        
        std::vector<double const *> chains(num_chains);
        for(size_t chain=0; chain!=num_chains; ++chain)
        {
            chains[chain] = vector.data()+draws_per_chain*chain;
        }
        sample_size[column] = stan::analyze::compute_potential_scale_reduction(
            chains, draws_per_chain);
    }
    
    return sample_size;
}
