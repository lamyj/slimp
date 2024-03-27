// WARNING: including <Eigen/Dense> (from ArrayWriter) before Stan headers
// creates compilation errors, possibly due to a template instantiated too
// early.
// #include "actions.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/standalone_gqs.hpp>

#include "action_parameters.h"
#include "ArrayWriter.h"
#include "Factory.h"
#include "VarContext.h"

std::tuple<ArrayWriter::Array, std::vector<std::string>> sample(
    std::string const & name, pybind11::dict data)
{
    // FIXME: move to function parameters
    action_parameters::Sample params;
    params.seed = 42;
    params.num_chains = 4;
    
    VarContext var_context(data);
    
    auto & model = Factory::instance().get(
        name+"_sampler", var_context, params.seed, &std::cout);
    
    std::vector<std::string> model_parameters;
    model.constrained_param_names(model_parameters);
    
    // Get the columns added by the sampler (e.g. lp__, treedepth__, etc.)
    std::vector<std::string> hmc_names;
    stan::mcmc::sample::get_sample_param_names(hmc_names);
    auto rng = stan::services::util::create_rng(0, 1);
    stan::mcmc::adapt_diag_e_nuts<decltype(model), decltype(rng)> sampler(model, rng);
    sampler.get_sampler_param_names(hmc_names);
    auto const hmc_fixed_cols = hmc_names.size();
    
    stan::callbacks::interrupt interrupt;
    // FIXME: return this
    std::vector<std::ostringstream> log_streams(5);
    stan::callbacks::stream_logger logger(
        log_streams[0], log_streams[1], log_streams[2], log_streams[3],
        log_streams[4]);
    
    std::vector<std::shared_ptr<stan::io::var_context>> init_contexts;
    std::vector<stan::callbacks::writer> init_writers(params.num_chains);
    ArrayWriter::Array sample_array({
        params.num_chains,
        size_t(
            params.save_warmup
            ?(params.num_warmup+params.num_samples)
            :params.num_samples),
        2 + hmc_names.size() + model_parameters.size()});
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
    // ArrayWriter::Array diagnostic_array({
    //     params.num_chains, size_t(params.num_warmup+params.num_samples), 
    //     hmc_names.size() + 3*model.num_params_r()});
    std::vector<ArrayWriter> sample_writers/* , diagnostic_writers */;
    std::vector<stan::callbacks::writer> diagnostic_writers(params.num_chains);
    for(size_t i=0; i!=params.num_chains; ++i)
    {
        init_contexts.push_back(
            std::make_shared<stan::io::empty_var_context>());
        sample_writers.emplace_back(sample_array, i, 2);
        // diagnostic_writers.emplace_back(diagnostic_array, i);
    }
    
    auto const return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
        model, params.num_chains, init_contexts, params.seed,
        params.id, params.init_radius, params.num_warmup, params.num_samples,
        params.thin, params.save_warmup, 0, params.hmc.stepsize,
        params.hmc.stepsize_jitter, params.hmc.max_depth, params.adapt.delta,
        params.adapt.gamma, params.adapt.kappa, params.adapt.t0,
        params.adapt.init_buffer, params.adapt.term_buffer, params.adapt.window,
        interrupt, logger, init_writers, sample_writers, diagnostic_writers);
    if(return_code != 0)
    {
        throw std::runtime_error(
            "Error while sampling: "+std::to_string(return_code));
    }
    
    auto names = sample_writers[0].names();
    names.insert(names.begin(), {"chain__", "draw__"});
    return std::make_tuple(sample_array, names);
}

std::tuple<ArrayWriter::Array, std::vector<std::string>> generate_quantities(
    std::string const & name, std::string const & variant,
    pybind11::dict data, Eigen::Ref<Eigen::MatrixXd> draws)
{
    // FIXME: move to function parameters
    action_parameters::Sample params;
    params.seed = 42;
    params.num_chains = 4;
    
    VarContext var_context(data);
    
    auto & model = Factory::instance().get(
        name+"_"+variant, var_context, params.seed, &std::cout);
    
    stan::callbacks::interrupt interrupt;
    // FIXME: return this
    std::vector<std::ostringstream> log_streams(5);
    stan::callbacks::stream_logger logger(
        log_streams[0], log_streams[1], log_streams[2], log_streams[3],
        log_streams[4]);
    
    auto const num_draws = draws.rows() / params.num_chains;
    
    std::vector<std::string> parameters;
    model.constrained_param_names(parameters, false, false);
    std::vector<std::string> generated_quantities;
    model.constrained_param_names(generated_quantities, false, true);
    auto const columns = generated_quantities.size() - parameters.size();
    
    ArrayWriter::Array array({params.num_chains, num_draws, 2+columns});
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
    for(size_t i=0; i!=params.num_chains; ++i)
    {
        draws_array.push_back(draws.block(i*num_draws, 0, num_draws, draws.cols()));
        writers.emplace_back(array, i, 2, parameters.size());
    }
    
    auto const return_code = stan::services::standalone_generate(
        model, params.num_chains, draws_array, params.seed, interrupt, logger,
        writers);
    if(return_code != 0)
    {
        throw std::runtime_error(
            "Error while sampling: "+std::to_string(return_code));
    }
    
    auto names = writers[0].names();
    names.insert(names.begin(), {"chain__", "draw__"});
    return std::make_tuple(array, names);
}
