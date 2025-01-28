#ifndef _eb77cafa_e85b_4b8c_b57b_cb9bbabab4c6
#define _eb77cafa_e85b_4b8c_b57b_cb9bbabab4c6

#include <string>
#include <vector>

#include <stan/io/var_context.hpp>
// #include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <xtensor/xtensor.hpp>

#include "slimp/action_parameters.h"

namespace slimp
{

template<typename T>
class Model
{
public:
    using Array = xt::xtensor<double, 3>;
    
    Model(
        stan::io::var_context & context,
        action_parameters::Sample const & parameters);
    
    std::vector<std::string> model_names(
        bool transformed_parameters=true, bool generated_quantities=true) const;
    std::vector<std::string> hmc_names() const;
    Array create_samples();
    void sample(Array & array);
    
    Array create_generated_quantities(Array const & draws);
    void generate(Array const & draws, Array & generated_quantities);
    
private:
    T _model;
    action_parameters::Sample _parameters;
};

}

#include "Model.txx"

#endif // _eb77cafa_e85b_4b8c_b57b_cb9bbabab4c6
