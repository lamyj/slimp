#ifndef _e32a55c4_7716_4457_8494_3bfea83f498e
#define _e32a55c4_7716_4457_8494_3bfea83f498e

#include "actions.h"

#include <string>
#include <vector>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "slimp/action_parameters.h"
#include "slimp/Model.h"
#include "slimp/VarContext.h"

namespace slimp
{

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

template<typename T>
pybind11::dict sample(
    pybind11::dict data, action_parameters::Sample const & parameters)
{
    stan::math::init_threadpool_tbb(
        parameters.sequential_chains
        ? parameters.threads_per_chain
        : parameters.num_chains*parameters.threads_per_chain);
    
    auto context = to_context(data);
    Model<T> model(context, parameters);
    auto samples = model.create_samples();
    model.sample(samples);
    
    std::vector<std::string> names = model.hmc_names();
    auto const model_names = model.model_names();
    std::copy(model_names.begin(), model_names.end(), std::back_inserter(names));
    
    auto const parameters_names = model.model_names(false, false);
    
    pybind11::dict result;
    result["array"] = samples;
    result["columns"] = names;
    result["parameters_columns"] = parameters_names;
    
    return result;
}

template<typename T>
pybind11::dict generate_quantities(
    pybind11::dict data, xt::xtensor<double, 3> const & draws,
    action_parameters::Sample const & parameters)
{
    auto context = to_context(data);
    Model<T> model(context, parameters);
    auto generated_quantities = model.create_generated_quantities(draws);
    model.generate(draws, generated_quantities);
    
    auto const model_names = model.model_names(true, true);
    std::vector<std::string> names{
        model_names.begin()+model.model_names(true, false).size(),
        model_names.end()};
    
    pybind11::dict result;
    result["array"] = generated_quantities;
    result["columns"] = names;
    
    return result;
}

template<typename Model>
void parallel_sample(
    slimp::VarContext const & context,
    slimp::action_parameters::Sample parameters, std::size_t R,
    ContextUpdater const & update_context,
    ResultsUpdater const & update_results)
{
    // NOTE: force sequential chains so that parallelization can take place at
    // the voxel level
    parameters.sequential_chains = true;
    
    oneapi::tbb::parallel_for(0UL, R, [&] (size_t r) {
        auto context_ = context;
        update_context(context_, r);
        
        Model model(context_, parameters);
        auto samples = model.create_samples();
        model.sample(samples, stan::callbacks::logger());
        
        update_results(samples, r);
    });
}

}

#endif // _e32a55c4_7716_4457_8494_3bfea83f498e
