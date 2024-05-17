# slimp: linear models with Stan and Pandas

*slimp* estimates linear models using [Stan](https://mc-stan.org/) and [Pandas](https://pandas.pydata.org/). Think [rstanarm](https://mc-stan.org/rstanarm/) or [brms](https://mc-stan.org/users/interfaces/brms), but in Python and faster.

Create the model:

```python
import matplotlib.pyplot
import numpy
import pandas
import slimp

y, x = numpy.mgrid[0:10, 0:10]
z = 10 + x + 2*y + numpy.random.normal(0, 2, (10, 10))
data = pandas.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": z.ravel()})

model = slimp.Model("z ~ 1 + x + y", data, num_chains=4)
# Also possible to specify random seed
# model = slimp.Model("z ~ 1 + x + y", data, seed=42)
```

Sample the parameters, check the results:

```python
model.sample()
print(model.hmc_diagnostics)
print(model.summary()[["N_Eff", "R_hat"]].describe().loc[["min", "max"], :])
r_squared = slimp.r_squared(model)
print(r_squared.quantile([0.05, 0.95]))
```

Plot prior and posterior predictive checks:

```python
figure, plots = matplotlib.pyplot.subplots(1, 2, layout="tight", figsize=(8, 4))
slimp.predictive_plot(model, use_prior=True, plot_kwargs={"ax":plots[0]})
slimp.predictive_plot(model, use_prior=False, plot_kwargs={"ax":plots[1]})
```

Plot the credible intervals of the parameters and their distributions:

```
slimp.parameters_plot(model, include=["x", "y"])
slimp.KDEPlot(model.draws["sigma"], prob=0.90)
```

Use a custom Stan model: have a look [here](custom_model_example/README.md)
