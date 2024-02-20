import matplotlib.pyplot
import numpy
import seaborn

def parameters_plot(model, include=None, exclude=None, **kwargs):
    if include is None:
        include = model.draws.columns
    if exclude is None:
        exclude = []
    
    columns = [
        x for x in model.draws.columns
        if x in include and x not in exclude]
    
    kwargs.setdefault("estimator", numpy.median)
    kwargs.setdefault("errorbar", ("pi", 90))
    
    ax = seaborn.pointplot(
        model.draws[columns].melt(),
        x="value", y="variable",
        linestyle="none",
        **kwargs)
    ax.set(xlabel=model.outcomes.columns[0], ylabel=None)

def predictive_plot(
        model, use_prior=False, predict_kwargs={}, count=50, alpha=0.2,
        plot_kwargs={}):
    _, y_posterior = model.predict(model.data, use_prior, **predict_kwargs)
    if "seed" in predict_kwargs:
        numpy.random.seed(predict_kwargs["seed"])
    subset = numpy.random.randint(0, len(y_posterior), count)
    
    for draw in subset:
        seaborn.kdeplot(
            y_posterior.iloc[draw, :], color="C0", alpha=alpha, **plot_kwargs)
    
    seaborn.kdeplot(
        model.outcomes.values.squeeze(), color="k", alpha=1, **plot_kwargs)
    plot_kwargs.get("ax", matplotlib.pyplot.gca()).set(
        xlabel=model.outcomes.columns[0])
