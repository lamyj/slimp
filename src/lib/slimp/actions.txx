#ifndef _e32a55c4_7716_4457_8494_3bfea83f498e
#define _e32a55c4_7716_4457_8494_3bfea83f498e

#include "actions.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/sample.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/services/util/create_rng.hpp>

#include "slimp/action_parameters.h"
#include "slimp/ArrayWriter.h"
#include "slimp/Logger.h"
#include "slimp/VarContext.h"

template<typename Model>
pybind11::dict sample(
    pybind11::dict data, action_parameters::Sample const & parameters)
{
    VarContext var_context(data);
    
    Model model(var_context, parameters.seed, &std::cout);
    
    std::vector<std::string> model_names;
    model.constrained_param_names(model_names);
    
    // Get the columns added by the sampler (e.g. lp__, treedepth__, etc.)
    std::vector<std::string> hmc_names;
    stan::mcmc::sample::get_sample_param_names(hmc_names);
    auto rng = stan::services::util::create_rng(0, 1);
    stan::mcmc::adapt_diag_e_nuts<decltype(model), decltype(rng)> sampler(
        model, rng);
    sampler.get_sampler_param_names(hmc_names);
    auto const hmc_fixed_cols = hmc_names.size();
    
    stan::callbacks::interrupt interrupt;
    Logger logger;
    
    std::vector<std::shared_ptr<stan::io::var_context>> init_contexts;
    for(size_t i=0; i!=parameters.num_chains; ++i)
    {
        init_contexts.push_back(
            std::make_shared<stan::io::empty_var_context>());
    }
    
    std::vector<stan::callbacks::writer> init_writers(parameters.num_chains);
    
    ArrayWriter::Array sample_array({
        parameters.num_chains,
        size_t(
            parameters.save_warmup
            ?(parameters.num_warmup+parameters.num_samples)
            :parameters.num_samples),
        2 + hmc_names.size() + model_names.size()});
    {
        auto && accessor = sample_array.mutable_unchecked();
        for(size_t chain=0; chain!=sample_array.shape(0); ++chain)
        {
            for(size_t sample=0; sample!=sample_array.shape(1); ++sample)
            {
                *accessor.mutable_data(chain, sample, 0UL) = 1+chain;
                *accessor.mutable_data(chain, sample, 1UL) = sample;
            }
        }
    }
    
    std::vector<ArrayWriter> sample_writers;
    for(size_t i=0; i!=parameters.num_chains; ++i)
    {
        // init_contexts.push_back(
        //     std::make_shared<stan::io::empty_var_context>());
        sample_writers.emplace_back(sample_array, i, 2);
    }
    
    std::vector<stan::callbacks::writer> diagnostic_writers(
        parameters.num_chains);
    
    auto const return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
        model, parameters.num_chains, init_contexts, parameters.seed,
        parameters.id, parameters.init_radius, parameters.num_warmup,
        parameters.num_samples, parameters.thin, parameters.save_warmup,
        parameters.refresh, parameters.hmc.stepsize,
        parameters.hmc.stepsize_jitter, parameters.hmc.max_depth,
        parameters.adapt.delta, parameters.adapt.gamma, parameters.adapt.kappa,
        parameters.adapt.t0, parameters.adapt.init_buffer,
        parameters.adapt.term_buffer, parameters.adapt.window, interrupt,
        logger, init_writers, sample_writers, diagnostic_writers);
    if(return_code != 0)
    {
        throw std::runtime_error(
            "Error while sampling: "+std::to_string(return_code));
    }
    
    auto names = sample_writers[0].names();
    names.insert(names.begin(), {"chain__", "draw__"});
    
    std::vector<std::string> parameters_names;
    model.constrained_param_names(parameters_names, false, false);
    
    pybind11::dict result;
    result["array"] = sample_array;
    result["columns"] = names;
    result["parameters_columns"] = parameters_names;
    
    return result;
}

template<typename Model>
pybind11::dict generate_quantities(
    pybind11::dict data, Eigen::Ref<Eigen::MatrixXd> draws,
    action_parameters::GenerateQuantities const & parameters)
{
    VarContext var_context(data);
    
    Model model(var_context, parameters.seed, &std::cout);
    
    stan::callbacks::interrupt interrupt;
    Logger logger;
    
    auto const num_draws = draws.rows() / parameters.num_chains;
    
    std::vector<std::string> model_names;
    model.constrained_param_names(model_names, false, false);
    std::vector<std::string> gq_names;
    model.constrained_param_names(gq_names, false, true);
    auto const columns = gq_names.size() - model_names.size();
    
    ArrayWriter::Array array({parameters.num_chains, num_draws, 2+columns});
    {
        auto && accessor = array.mutable_unchecked();
        for(size_t chain=0; chain!=array.shape(0); ++chain)
        {
            for(size_t sample=0; sample!=array.shape(1); ++sample)
            {
                *accessor.mutable_data(chain, sample, 0UL) = 1+chain;
                *accessor.mutable_data(chain, sample, 1UL) = sample;
            }
        }
    }
        
    // FIXME: are the draws copied in draws_array?
    std::vector<Eigen::MatrixXd> draws_array;
    std::vector<ArrayWriter> writers;
    for(size_t i=0; i!=parameters.num_chains; ++i)
    {
        draws_array.push_back(
            draws.block(i*num_draws, 0, num_draws, draws.cols()));
        writers.emplace_back(array, i, 2, model_names.size());
    }
    
    auto const return_code = stan::services::standalone_generate(
        model, parameters.num_chains, draws_array, parameters.seed, interrupt,
        logger, writers);
    if(return_code != 0)
    {
        throw std::runtime_error(
            "Error while sampling: "+std::to_string(return_code));
    }
    
    auto names = writers[0].names();
    names.insert(names.begin(), {"chain__", "draw__"});
    pybind11::dict result;
    result["array"] = array;
    result["columns"] = names;
    
    return result;
}


#endif // _e32a55c4_7716_4457_8494_3bfea83f498e
