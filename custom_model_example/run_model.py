import matplotlib.pyplot
import numpy
import pandas
import slimp

import my_model

y, x = numpy.mgrid[0:10, 0:10]
z = 10 + x + 2*y + numpy.random.normal(0, 2, (10, 10))

data = pandas.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": z.ravel()})

fit_data = {
    "N": len(data),
    
    "x": data["x"], "y": data["y"], "z": data["z"]
}

sampler_parameters = slimp.action_parameters.Sample(num_chains=4, seed=42)
samples = slimp.sample_data_as_df(my_model.sample(fit_data, sampler_parameters))

print(slimp.stats.hmc_diagnostics(samples, sampler_parameters.hmc.max_depth))

draws = samples.filter(["a", "b_x", "b_y", "sigma"])
summary = slimp.summary(
    samples[["lp__"]].join(draws), sampler_parameters.num_chains)
print(summary)

generate_parameters = slimp.action_parameters.GenerateQuantities(
    seed=sampler_parameters.seed, num_chains=sampler_parameters.num_chains)
        
generated = slimp.sample_data_as_df(
    my_model.generate_quantities(fit_data, draws.values, generate_parameters))
