#ifndef _e32a55c4_7716_4457_8494_3bfea83f498e
#define _e32a55c4_7716_4457_8494_3bfea83f498e

#include "actions.h"

#include <string>
#include <vector>

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/pybind11.h>

#include "slimp/action_parameters.h"
#include "slimp/Model.h"
#include "slimp/VarContext.h"

namespace slimp
{

template<typename T>
pybind11::dict sample(
    pybind11::dict data, action_parameters::Sample const & parameters)
{
    auto const g = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, 
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
    std::function<void(VarContext &, std::size_t)> const & update_context,
    ResultsUpdater const & update_results)
{
    // NOTE: force sequential chains so that parallelization can take place at
    // the voxel level
    parameters.sequential_chains = true;
    
    oneapi::tbb::enumerable_thread_specific<slimp::VarContext> context_(context);
    oneapi::tbb::parallel_for(0UL, R, [&] (size_t r) {
        update_context(context_.local(), r);
        
        Model model(context_.local(), parameters);
        auto samples = model.create_samples();
        model.sample(samples, stan::callbacks::logger());
        
        update_results(samples, r);
    });
}

template<typename Model>
void parallel_sample(
    slimp::VarContext const & context,
    slimp::action_parameters::Sample parameters, std::size_t R,
    std::function<bool(VarContext &, std::size_t)> const & update_context,
    ResultsUpdater const & update_results)
{
    // NOTE: force sequential chains so that parallelization can take place at
    // the voxel level
    parameters.sequential_chains = true;
    
    oneapi::tbb::enumerable_thread_specific<slimp::VarContext> context_(context);
    oneapi::tbb::parallel_for(0UL, R, [&] (size_t r) {
        auto const may_run = update_context(context_.local(), r);
        
        if(!may_run)
        {
            return;
        }
        
        Model model(context_.local(), parameters);
        auto samples = model.create_samples();
        model.sample(samples, stan::callbacks::logger());
        
        update_results(samples, r);
    });
}

}

#endif // _e32a55c4_7716_4457_8494_3bfea83f498e
