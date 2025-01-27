#ifndef _401b1db3_bc8e_4f90_9c04_3d877467ab5c
#define _401b1db3_bc8e_4f90_9c04_3d877467ab5c

#include "Model.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stan/callbacks/interrupt.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/standalone_gqs.hpp>

#include <xtensor/xmanipulation.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "slimp/action_parameters.h"
#include "slimp/ArrayWriter.h"
#include "slimp/Logger.h"

namespace slimp
{

template<typename T>
Model<T>
::Model(
    stan::io::var_context & context,
    action_parameters::Sample const & parameters)
: _model(context, parameters.seed, &std::cout), _parameters(parameters),
    _rng(stan::services::util::create_rng<RandomNumberGenerator>(0, 1)),
    _sampler(this->_model, this->_rng)
{
    // Nothing else.
}

template<typename T>
std::vector<std::string>
Model<T>
::model_names(bool transformed_parameters, bool generated_quantities) const
{
    std::vector<std::string> model_names;
    this->_model.constrained_param_names(
        model_names, transformed_parameters, generated_quantities);
    return model_names;
}

template<typename T>
std::vector<std::string>
Model<T>
::hmc_names() const
{
    std::vector<std::string> hmc_names;
    stan::mcmc::sample::get_sample_param_names(hmc_names);
    this->_sampler.get_sampler_param_names(hmc_names);
    return hmc_names;
}

template<typename T>
typename Model<T>::Array
Model<T>
::create_samples()
{
    Array array(Array::shape_type{
        2 + this->hmc_names().size() + this->model_names().size(),
        this->_parameters.num_chains,
        size_t(
            this->_parameters.save_warmup
            ?(this->_parameters.num_warmup+this->_parameters.num_samples)
            :this->_parameters.num_samples)});
    
    xt::ravel(xt::view(array, 0)) = xt::repeat(
        xt::eval(xt::arange(1UL, 1+array.shape(1))), array.shape(2), 0);
    xt::ravel(xt::view(array, 1)) = xt::tile(
        xt::arange(0UL, array.shape(2)), array.shape(1));
    
    return array;
}

template<typename T>
void
Model<T>
::sample(Array & array)
{
    stan::callbacks::interrupt interrupt;
    Logger logger;
    
    std::vector<std::shared_ptr<stan::io::var_context>> init_contexts;
    auto const & parameters = this->_parameters;
    auto const & num_chains = parameters.num_chains;
    for(size_t i=0; i!=num_chains; ++i)
    {
        init_contexts.push_back(std::make_shared<stan::io::empty_var_context>());
    }
    
    std::vector<stan::callbacks::writer> init_writers(num_chains);
    
    std::vector<ArrayWriter> sample_writers;
    for(size_t i=0; i!=num_chains; ++i)
    {
        sample_writers.emplace_back(array, i, 2UL);
    }
    
    std::vector<stan::callbacks::writer> diagnostic_writers(num_chains);
    
    if(parameters.sequential_chains)
    {
        for(std::size_t chain=0; chain!=num_chains; ++chain)
        {
            auto const return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
                this->_model, *init_contexts[chain], parameters.seed,
                chain, parameters.init_radius, parameters.num_warmup,
                parameters.num_samples, parameters.thin, parameters.save_warmup,
                parameters.refresh, parameters.hmc.stepsize,
                parameters.hmc.stepsize_jitter, parameters.hmc.max_depth,
                parameters.adapt.delta, parameters.adapt.gamma, parameters.adapt.kappa,
                parameters.adapt.t0, parameters.adapt.init_buffer,
                parameters.adapt.term_buffer, parameters.adapt.window, interrupt,
                logger, init_writers[chain], sample_writers[chain], diagnostic_writers[chain]);
            if(return_code != 0)
            {
                throw std::runtime_error(
                    "Error while sampling: "+std::to_string(return_code));
            }
        }
    }
    else
    {
        auto const return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
            this->_model, num_chains, init_contexts, parameters.seed,
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
    }
}

template<typename T>
typename Model<T>::Array
Model<T>
::create_generated_quantities()
{
    std::vector<std::string> model_names;
    this->_model.constrained_param_names(model_names, false, false);
    std::vector<std::string> gq_names;
    this->_model.constrained_param_names(gq_names, false, true);
    auto const parameters = gq_names.size() - model_names.size();
    
    Array array(Array::shape_type{
        2 + parameters,
        this->_parameters.num_chains,
        size_t(
            this->_parameters.save_warmup
            ?(this->_parameters.num_warmup+this->_parameters.num_samples)
            :this->_parameters.num_samples)});
    
    xt::ravel(xt::view(array, 0)) = xt::repeat(
        xt::eval(xt::arange(1UL, 1+array.shape(0))), array.shape(1), 0);
    xt::ravel(xt::view(array, 1)) = xt::tile(
        xt::arange(0UL, array.shape(1)), array.shape(0));
    
    return array;
}

template<typename T>
void
Model<T>
::generate(Array const & draws, Array & generated_quantities)
{
    stan::callbacks::interrupt interrupt;
    Logger logger;
    
    std::vector<std::string> model_names;
    this->_model.constrained_param_names(model_names, false, false);
    
    // WARNING: the copy to draws_array is mandated by the Stan API (not
    // possible to use Eigen::Ref or Eigen::Map)
    std::vector<Eigen::MatrixXd> draws_array;
    std::vector<ArrayWriter> writers;
    for(size_t chain=0; chain!=draws.shape(1); ++chain)
    {
        Eigen::MatrixXd destination(draws.shape(2), draws.shape(0));
        for(std::size_t parameter=0; parameter!=draws.shape(0); ++parameter)
        {
            for(std::size_t draw=0; draw!=draws.shape(2); ++draw)
            {
                destination(draw, parameter) = draws(parameter, chain, draw);
            }
        }
        writers.emplace_back(
            generated_quantities, chain, 2UL, model_names.size());
    }
    
    auto const return_code = stan::services::standalone_generate(
        this->_model, draws.shape(1), draws_array, this->_parameters.seed,
        interrupt, logger, writers);
    if(return_code != 0)
    {
        throw std::runtime_error(
            "Error while sampling: "+std::to_string(return_code));
    }
}

}

#endif // _401b1db3_bc8e_4f90_9c04_3d877467ab5c
