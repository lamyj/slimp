# slimp: linear models with Stan and Pandas

```python
import numpy
import pandas
import slimp

data = pandas.DataFrame({
    "X1": numpy.linspace(-5, 10, 100),
    "X2": numpy.linspace(0, -1, 100)
})
data["y"] = (
    3 + 3*data["X1"] + 10 * data["X2"]
    + numpy.random.normal(0, 1, len(data)))

model = slimp.Model("y ~ 1 + X1 + X2", data)
model.sample(seed=42, chains=4, parallel_chains=4, show_progress=False)

# A couple of maximum tree dephths but nothing bad
print(model.hmc_diagnostics)

# Good exploration
print(model.summary()[["N_Eff", "R_hat"]].describe().loc[["min", "max"], :])

# Good R²
r_squared = slimp.r_squared(model)
print(r_squared.describe())
```
