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
#include <pybind11/stl.h>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/standalone_gqs.hpp>

#include "action_parameters.h"
#include "ArrayWriter.h"
#include "Factory.h"
#include "VarContext.h"

pybind11::dict sample(
    std::string const & name, pybind11::dict data,
    action_parameters::Sample const & parameters)
{
    VarContext var_context(data);
    
    auto & model = Factory::instance().get(
        name+"_sampler", var_context, parameters.seed, &std::cout);
    
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
    // FIXME: return this
    std::vector<std::ostringstream> log_streams(5);
    stan::callbacks::stream_logger logger(
        log_streams[0], log_streams[1], log_streams[2], log_streams[3],
        log_streams[4]);
    
    std::vector<std::shared_ptr<stan::io::var_context>> init_contexts;
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
    // ArrayWriter::Array diagnostic_array({
    //     params.num_chains, size_t(params.num_warmup+params.num_samples), 
    //     hmc_names.size() + 3*model.num_params_r()});
    std::vector<ArrayWriter> sample_writers/* , diagnostic_writers */;
    std::vector<stan::callbacks::writer> diagnostic_writers(
        parameters.num_chains);
    for(size_t i=0; i!=parameters.num_chains; ++i)
    {
        init_contexts.push_back(
            std::make_shared<stan::io::empty_var_context>());
        sample_writers.emplace_back(sample_array, i, 2);
        // diagnostic_writers.emplace_back(diagnostic_array, i);
    }
    
    auto const return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
        model, parameters.num_chains, init_contexts, parameters.seed,
        parameters.id, parameters.init_radius, parameters.num_warmup,
        parameters.num_samples, parameters.thin, parameters.save_warmup, 0,
        parameters.hmc.stepsize, parameters.hmc.stepsize_jitter,
        parameters.hmc.max_depth, parameters.adapt.delta,
        parameters.adapt.gamma, parameters.adapt.kappa, parameters.adapt.t0,
        parameters.adapt.init_buffer, parameters.adapt.term_buffer,
        parameters.adapt.window, interrupt, logger, init_writers,
        sample_writers, diagnostic_writers);
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

pybind11::dict generate_quantities(
    std::string const & name, std::string const & variant,
    pybind11::dict data, Eigen::Ref<Eigen::MatrixXd> draws,
    action_parameters::GenerateQuantities const & parameters)
{
    VarContext var_context(data);
    
    auto & model = Factory::instance().get(
        name+"_"+variant, var_context, parameters.seed, &std::cout);
    
    stan::callbacks::interrupt interrupt;
    // FIXME: return this
    std::vector<std::ostringstream> log_streams(5);
    stan::callbacks::stream_logger logger(
        log_streams[0], log_streams[1], log_streams[2], log_streams[3],
        log_streams[4]);
    
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
