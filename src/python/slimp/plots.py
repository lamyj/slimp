import matplotlib.pyplot
import numpy
import seaborn
import scipy.stats

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

def predictive_plot(model, use_prior=False, count=50, alpha=0.2, plot_kwargs={}):
    y = model.prior_predict if use_prior else model.posterior_predict
    subset = numpy.random.randint(0, len(y), count)
    
    for outcome_index in range(len(model.outcomes.columns)):
        y_outcome = (
            y.filter(regex=f"\\.{1+outcome_index}$")
            if len(model.outcomes.columns) > 1 else y)
        color = f"C{outcome_index}" if len(model.outcomes.columns) > 1 else "k"
        for draw_index, draw in enumerate(subset):
            seaborn.kdeplot(
                y_outcome.values[draw, :], color=color,
                alpha=alpha, **plot_kwargs)
        seaborn.kdeplot(
            model.outcomes.iloc[:, outcome_index], color=color, alpha=1,
            **plot_kwargs)

class KDEPlot:
    def __init__(
            self, data, ax=None, prob=None, point_estimate=None, color="black",
            alpha=0.3, **kwargs):
        
        self.ax = ax or matplotlib.pyplot.gca()
        
        linewidth = kwargs.pop("lw", kwargs.pop("linewidth", None))
        
        self.kde = scipy.stats.gaussian_kde(data)
        bw = self.kde.scotts_factor() * data.std(ddof=1)
        
        xs = numpy.linspace(data.min()-bw, data.max()+bw, 200)
        self.dist_line = self.ax.plot(
            xs, self.kde(xs), color=color, linewidth=linewidth)
        
        self.dist_area = None
        if prob is not None:
            self.q = numpy.quantile(data, [0.5-prob/2, 0.5+prob/2])
            xs = xs[(xs>=self.q[0]) & (xs<=self.q[1])]
            self.dist_area = self.ax.fill_between(
                xs, self.kde(xs), color=color, alpha=alpha, lw=0)
        
        self.estimate_line = None
        if point_estimate is not None:
            x = point_estimate(data)
            self.estimate_line = self.ax.plot(
                [x, x], [0, *self.kde(x)], color=color, linewidth=linewidth)
        
        self.ax.set(ylim=0, yticks=[])
        self.ax.spines[["top", "left", "right"]].set_visible(False)
