#include <pybind11/pybind11.h>

// WARNING: Factory *must* be included first to avoid Eigen-related compilation
// errors
#include "Factory.h"

#include "action_parameters.h"
#include "actions.h"

#include "multivariate_log_likelihood.h"
#include "multivariate_predict_posterior.h"
#include "multivariate_predict_prior.h"
#include "multivariate_sampler.h"

#include "univariate_log_likelihood.h"
#include "univariate_predict_posterior.h"
#include "univariate_predict_prior.h"
#include "univariate_sampler.h"

#define REGISTER(prefix, suffix) \
    Factory::instance().register_(#prefix "_" #suffix, new_##prefix##_##suffix);

#define REGISTER_ALL(name) \
    REGISTER(name, sampler); \
    REGISTER(name, log_likelihood); \
    REGISTER(name, predict_prior); \
    REGISTER(name, predict_posterior);

PYBIND11_MODULE(_slimp, module)
{
    REGISTER_ALL(univariate);
    REGISTER_ALL(multivariate);
    
    auto action_parameters_ = module.def_submodule("action_parameters");
    
    auto adapt_pickler = std::make_pair(
        [](action_parameters::Adapt const & self){
            pybind11::dict state;
            
            state["engaged"] = self.engaged;
            state["gamma"] = self.gamma;
            state["delta"] = self.delta;
            state["kappa"] = self.kappa;
            state["t0"] = self.t0;
            state["init_buffer"] = self.init_buffer;
            state["term_buffer"] = self.term_buffer;
            state["window"] = self.window;
            state["save_metric"] = self.save_metric;
            
            return state;
        },
        [](pybind11::dict state) {
            action_parameters::Adapt self;
            
            self.engaged = state["engaged"].cast<bool>();
            self.gamma = state["gamma"].cast<double>();
            self.delta = state["delta"].cast<double>();
            self.kappa = state["kappa"].cast<double>();
            self.t0 = state["t0"].cast<double>();
            self.init_buffer = state["init_buffer"].cast<unsigned int>();
            self.term_buffer = state["term_buffer"].cast<unsigned int>();
            self.window = state["window"].cast<unsigned int>();
            self.save_metric = state["save_metric"].cast<bool>();
            
            return self;
        });
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
        .def_readwrite("save_metric", &action_parameters::Adapt::save_metric)
        .def(pybind11::pickle(adapt_pickler.first, adapt_pickler.second));
    
    auto const hmc_pickler = std::make_pair(
        [](action_parameters::HMC const & self){
            pybind11::dict state;
            
            state["int_time"] = self.int_time;
            state["max_depth"] = self.max_depth;
            state["stepsize"] = self.stepsize;
            state["stepsize_jitter"] = self.stepsize_jitter;
            
            return state;
        },
        [](pybind11::dict state){
            action_parameters::HMC self;
            
            self.int_time = state["int_time"].cast<double>();
            self.max_depth = state["max_depth"].cast<int>();
            self.stepsize = state["stepsize"].cast<double>();
            self.stepsize_jitter = state["stepsize_jitter"].cast<double>();
            
            return self;
        });
    pybind11::class_<action_parameters::HMC>(action_parameters_, "HMC")
        .def(pybind11::init<>())
        .def_readwrite("int_time", &action_parameters::HMC::int_time)
        .def_readwrite("max_depth", &action_parameters::HMC::max_depth)
        .def_readwrite("stepsize", &action_parameters::HMC::stepsize)
        .def_readwrite(
            "stepsize_jitter", &action_parameters::HMC::stepsize_jitter)
        .def(pybind11::pickle(hmc_pickler.first, hmc_pickler.second));
    
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
        .def_readwrite("init_radius", &action_parameters::Sample::init_radius)
        .def(pybind11::pickle(
            [&](action_parameters::Sample const & self) {
                pybind11::dict state;
                
                state["num_samples"] = self.num_samples;
                state["num_warmup"] = self.num_warmup;
                state["save_warmup"] = self.save_warmup;
                state["thin"] = self.thin;
                state["adapt"] = adapt_pickler.first(self.adapt);
                state["hmc"] = hmc_pickler.first(self.hmc);
                state["num_chains"] = self.num_chains;
                state["seed"] = self.seed;
                state["id"] = self.id;
                state["init_radius"] = self.init_radius;
                
                return state;
            },
            [&](pybind11::dict const & state) {
                action_parameters::Sample self;
                
                self.num_samples = state["num_samples"].cast<int>();
                self.num_warmup = state["num_warmup"].cast<int>();
                self.save_warmup = state["save_warmup"].cast<bool>();
                self.thin = state["thin"].cast<int>();
                self.adapt = adapt_pickler.second(state["adapt"]);
                self.hmc = hmc_pickler.second(state["hmc"]);
                self.num_chains = state["num_chains"].cast<size_t>();
                self.seed = state["seed"].cast<long>();
                self.id = state["id"].cast<int>();
                self.init_radius = state["init_radius"].cast<double>();
                
                return self;
            }));
    
    pybind11::class_<action_parameters::GenerateQuantities>(
            action_parameters_, "GenerateQuantities")
        .def(pybind11::init<>())
        .def_readwrite(
            "num_chains", &action_parameters::GenerateQuantities::num_chains)
        .def_readwrite("seed", &action_parameters::GenerateQuantities::seed);
    
    module.def("generate_quantities", &generate_quantities);
    module.def("sample", &sample);
    module.def("get_effective_sample_size", &get_effective_sample_size);
    module.def("get_potential_scale_reduction", &get_potential_scale_reduction);
}
