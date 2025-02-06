// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "slimp/actions.h"

#include "my_model.h"

PYBIND11_MODULE(my_model, module)
{
    module.def("sample", &slimp::sample<my_model::model>);
    module.def(
        "generate_quantities", &slimp::generate_quantities<my_model::model>);
}
