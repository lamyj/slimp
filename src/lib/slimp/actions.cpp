#include "actions.h"

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/pybind11.h>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <xtensor-python/pyarray.hpp>

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

xt::xtensor<double, 2> wrapper(
    xt::xtensor<double, 4> const & data,
    xt::xtensor<double, 1> (*function)(xt::xtensor<double, 3> const &))
{
    auto const num_threads_string = std::getenv("NUM_THREADS");
    std::size_t num_threads = 1;
    if(num_threads_string != nullptr)
    {
        try
        {
            num_threads = std::stoul(num_threads_string);
        }
        catch(std::exception &)
        {
            // Do nothing, keep the default value.
        }
    }
    auto const g = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, num_threads);
    xt::xtensor<double, 2> result(
        xt::xtensor<double, 2>::shape_type{data.shape()[0], data.shape()[1]});
    
    oneapi::tbb::parallel_for(0UL, data.shape()[0], [&] (size_t r) {
        xt::view(result, r) = function(xt::eval(xt::view(data, r)));
    });
    
    return result;
}

xt::xtensor<double, 2> get_effective_sample_size(
    xt::xtensor<double, 4> const & data)
{
    return wrapper(data, get_effective_sample_size);
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

xt::xtensor<double, 2> get_potential_scale_reduction(
    xt::xtensor<double, 4> const & data)
{
    return wrapper(data, get_potential_scale_reduction);
}

xt::xtensor<double, 1> get_split_potential_scale_reduction(
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
        R_hat[parameter] = stan::analyze::compute_split_potential_scale_reduction(
            chains, draws.shape(2));
    }
    
    return R_hat;
}

xt::xtensor<double, 2> get_split_potential_scale_reduction(
    xt::xtensor<double, 4> const & data)
{
    return wrapper(data, get_split_potential_scale_reduction);
}

VarContext to_context(pybind11::dict data)
{
    VarContext context;
    for(auto && item: data)
    {
        auto const & key = item.first.cast<std::string>();
        auto const & value = item.second;
        if(pybind11::isinstance<pybind11::int_>(value))
        {
            context.set(key, value.cast<int>());
        }
        else if(pybind11::isinstance<pybind11::float_>(value))
        {
            context.set(key, value.cast<double>());
        }
        else
        {
            // https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars-built-in
            auto const dtype = value.cast<pybind11::array>().dtype().char_();
            
            // Signed integer type
            if(dtype == 'b') 
            {
                context.set(key, value.cast<xt::xarray<int8_t>>());
            }
            else if(dtype == 'h')
            {
                context.set(key, value.cast<xt::xarray<int16_t>>());
            }
            else if(dtype == 'i')
            {
                context.set(key, value.cast<xt::xarray<int32_t>>());
            }
            else if(dtype == 'l')
            {
                context.set(key, value.cast<xt::xarray<int64_t>>());
            }
            // Unsigned integer types
            else if(dtype == 'B')
            {
                context.set(key, value.cast<xt::xarray<uint8_t>>());
            }
            else if(dtype == 'H')
            {
                context.set(key, value.cast<xt::xarray<uint16_t>>());
            }
            else if(dtype == 'I')
            {
                context.set(key, value.cast<xt::xarray<uint32_t>>());
            }
            else if(dtype == 'L')
            {
                context.set(key, value.cast<xt::xarray<uint64_t>>());
            }
            // Floating-point types
            else if(dtype == 'f')
            {
                context.set(key, value.cast<xt::xarray<float>>());
            }
            else if(dtype == 'd')
            {
                context.set(key, value.cast<xt::xarray<double>>());
            }
            // Unsupported type
            else
            {
                throw std::runtime_error(
                    std::string("Array type not handled: ")+ dtype);
            }
        }
    }
    
    return context;
}

}
