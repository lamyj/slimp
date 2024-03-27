#include <pybind11/pybind11.h>

// WARNING: Factory *must* be included first to avoid Eigen-related compilation
// errors
#include "Factory.h"

#include "action_parameters.h"
#include "actions.h"

#include "univariate_log_likelihood.h"
#include "univariate_predict_posterior.h"
#include "univariate_predict_prior.h"
#include "univariate_sampler.h"

#define REGISTER(name) \
    Factory::instance().register_(#name"_sampler", new_##name##_sampler); \
    Factory::instance().register_( \
        #name"_log_likelihood", new_##name##_log_likelihood); \
    Factory::instance().register_( \
        #name"_predict_prior", new_##name##_predict_prior); \
    Factory::instance().register_( \
        #name"_predict_posterior", new_##name##_predict_posterior);

PYBIND11_MODULE(_slimp, module)
{
    REGISTER(univariate);
    
    auto action_parameters_ = module.def_submodule("action_parameters");
    
    pybind11::class_<action_parameters::Adapt>(action_parameters_, "Adapt")
        .def(pybind11::init<>())
        .def_readwrite("engaged", &action_parameters::Adapt::engaged)
        .def_readwrite("gamma", &action_parameters::Adapt::gamma)
        .def_readwrite("delta", &action_parameters::Adapt::delta)
        .def_readwrite("kappa", &action_parameters::Adapt::kappa)
        .def_readwrite("t0", &action_parameters::Adapt::t0)
        .def_readwrite("init_buffer", &action_parameters::Adapt::init_buffer)
        .def_readwrite("term_buffer", &action_parameters::Adapt::term_buffer)
        .def_readwrite("window", &action_parameters::Adapt::window)
        .def_readwrite("save_metric", &action_parameters::Adapt::save_metric);
    
    pybind11::class_<action_parameters::HMC>(action_parameters_, "HMC")
        .def(pybind11::init<>())
        .def_readwrite("int_time", &action_parameters::HMC::int_time)
        .def_readwrite("max_depth", &action_parameters::HMC::max_depth)
        .def_readwrite("stepsize", &action_parameters::HMC::stepsize)
        .def_readwrite(
            "stepsize_jitter", &action_parameters::HMC::stepsize_jitter);
    
    pybind11::class_<action_parameters::Sample>(action_parameters_, "Sample")
        .def(pybind11::init<>())
        .def_readwrite("num_samples", &action_parameters::Sample::num_samples)
        .def_readwrite("num_warmup", &action_parameters::Sample::num_warmup)
        .def_readwrite("save_warmup", &action_parameters::Sample::save_warmup)
        .def_readwrite("thin", &action_parameters::Sample::thin)
        .def_readwrite("adapt", &action_parameters::Sample::adapt)
        .def_readwrite("hmc", &action_parameters::Sample::hmc)
        .def_readwrite("num_chains", &action_parameters::Sample::num_chains)
        .def_readwrite("seed", &action_parameters::Sample::seed)
        .def_readwrite("id", &action_parameters::Sample::id)
        .def_readwrite("init_radius", &action_parameters::Sample::init_radius);
    
    pybind11::class_<action_parameters::GenerateQuantities>(
            action_parameters_, "GenerateQuantities")
        .def(pybind11::init<>())
        .def_readwrite(
            "num_chains", &action_parameters::GenerateQuantities::num_chains)
        .def_readwrite("seed", &action_parameters::GenerateQuantities::seed);
    
    module.def("generate_quantities", &generate_quantities);
    module.def("sample", &sample);
}
