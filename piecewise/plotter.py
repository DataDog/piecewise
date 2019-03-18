# prj
from .regressor import piecewise


def piecewise_plot(t, v, min_stop_frac=0.03, model=None):
    """ Fits a piecewise (aka "segmented") regression and creates a scatter plot
    of the data overlaid with the regression segments.

    Params:
        t (listlike of ints or floats): independent/predictor variable values
        v (listlike of ints or floats): dependent/outcome variable values
        min_stop_frac (float between 0 and 1): the fraction of total error that
            a merge must account for to be considered "too big" to keep merging;
            the default is usually adequate, but this may be increased to make
            merging more aggressive (leading to fewer segments in the result)
        model (FittedModel): a previously fit model, if available
    Returns:
        None.
    """
    # 3p
    # delay importing until first use (enables export in __init__.py)
    import matplotlib.pyplot as plt
    
    model = model or piecewise(t, v, min_stop_frac)
    print('Num segments: %s' % len(model.segments))
    plt.plot(t, v, '.', alpha=0.6)
    for seg in model.segments:
        t_new = [seg.start_t, seg.end_t]
        v_hat = [seg.predict(t) for t in t_new]
        plt.plot(t_new, v_hat, 'k-')
    plt.show()

# alias for backward compatibility
plot_data_with_regression = piecewise_plot
