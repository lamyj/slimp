#include <pybind11/pybind11.h>

// WARNING: Factory *must* be included first to avoid Eigen-related compilation
// errors
#include "Factory.h"

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
    
    module.def("generate_quantities", &generate_quantities);
    module.def("sample", &sample);
}
