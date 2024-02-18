import matplotlib.pyplot
import numpy
import seaborn

def predictive_plot(
        model, use_prior=False, predict_kwargs={}, count=50, alpha=0.2,
        plot_kwargs={}):
    _, y_posterior = model.predict(model.data, use_prior)
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
