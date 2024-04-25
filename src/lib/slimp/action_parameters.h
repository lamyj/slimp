#ifndef _d498d353_df89_48aa_b410_419d66b6be60
#define _d498d353_df89_48aa_b410_419d66b6be60

#include <stddef.h>

#include "slimp/api.h"

namespace action_parameters
{

class SLIMP_API Adapt
{
public:
    bool engaged=true;
    double gamma = 0.05;
    double delta = 0.8;
    double kappa = 0.75;
    double t0 = 10;
    unsigned int init_buffer = 75;
    unsigned int term_buffer = 50;
    unsigned int window = 25;
    bool save_metric=false;
};

class SLIMP_API HMC
{
public:
    double int_time = 6.28319;
    int max_depth = 10;
    double stepsize = 1;
    double stepsize_jitter = 0;
};

class SLIMP_API Sample
{
public:
    int num_samples = 1000;
    int num_warmup = 1000;
    bool save_warmup = false;
    int thin = 1;
    Adapt adapt;
    HMC hmc;
    size_t num_chains = 1;
    long seed = -1;
    int id = 1;
    double init_radius = 2;
    int refresh = 0;
};

class SLIMP_API GenerateQuantities
{
public:
    size_t num_chains = 1;
    long seed = -1;
};

}

#endif // _d498d353_df89_48aa_b410_419d66b6be60
