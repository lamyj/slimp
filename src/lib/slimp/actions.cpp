#include "actions.h"

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/pybind11.h>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "slimp/api.h"
#include "slimp/action_parameters.h"

namespace slimp
{

xt::xtensor<double, 1> get_effective_sample_size(
    xt::xtensor<double, 3> const & draws)
{
    xt::xtensor<double, 1> sample_size(
        xt::xtensor<double, 1>::shape_type{draws.shape(0)});
    
    for(size_t parameter=0; parameter!=draws.shape(0); ++parameter)
    {
        // WARNING: this assumes that the draws array is C-contiguous
        std::vector<double const *> chains(draws.shape(1));
        for(size_t chain=0; chain!=chains.size(); ++chain)
        {
            chains[chain] = &draws.unchecked(parameter, chain);
        }
        sample_size[parameter] = stan::analyze::compute_effective_sample_size(
            chains, draws.shape(2));
    }
    
    return sample_size;
}

xt::xtensor<double, 1> get_potential_scale_reduction(
    xt::xtensor<double, 3> const & draws)
{
    xt::xtensor<double, 1> R_hat(
        xt::xtensor<double, 1>::shape_type{draws.shape(0)});
    for(size_t parameter=0; parameter!=draws.shape(0); ++parameter)
    {
        // WARNING: this assumes that the draws array is C-contiguous
        std::vector<double const *> chains(draws.shape(1));
        for(size_t chain=0; chain!=chains.size(); ++chain)
        {
            chains[chain] = &draws.unchecked(parameter, chain);
        }
        R_hat[parameter] = stan::analyze::compute_potential_scale_reduction(
            chains, draws.shape(2));
    }
    
    return R_hat;
}

}
