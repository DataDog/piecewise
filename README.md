# piecewise

This repo accompanies [Piecewise regression: when one line simply isn’t enough](https://www.datadoghq.com/blog/engineering/piecewise-regression/), a blog post about Datadog's approach to piecewise regression. The code included here is intended to be minimal and readable; this is not a Swiss Army knife to solve all variations of piecewise regression problems.

## Installation & dependencies

This package was written to work with both Python 2 and Python 3.

To install this package using setup tools, clone this repo and run `python setup.py install` from within the `piecewise` root directory.

The package's core `piecewise()` function for regression requires only `numpy`. The use of `piecewise_plot()` for plotting depends also on `matplotlib`.

## Usage

Start by preparing your data as list-likes of timestamps (independent variables) and values (dependent variables).

```
import numpy as np

t = np.arange(10)
v = np.array(
    [2*i for i in range(5)] +
    [10-i for i in range(5, 10)]
) + np.random.normal(0, 1, 10)
```

Now, you're ready to import the `piecewise()` function and fit a piecewise linear regression.

```
from piecewise import piecewise

model = piecewise(t, v)
```

`model` if a `FittedModel` object. If you are at a shell, you can print the object to see the fitted segments domains and regression coefficients.

```
>>> model
FittedModel with segments:
* FittedSegment(start_t=0, end_t=5, coeffs=(-0.8576123780622642, 2.224791099812951))
* FittedSegment(start_t=5, end_t=9, coeffs=(10.975487672814133, -1.0722348284390741))
```

Alternatively, you can use the `FittedModel`'s `segments` attribute to get at values.

```
>>> len(model.segments)
2
>>> model.segments[0].coeffs
(-0.8576123780622642, 2.224791099812951)
```

If you want to interpolate or extrapolate, you can use the `FittedModel`'s `predict()` function.

```
>>> model.predict(t_new=[3.5, 100])
array([  6.92915647, -96.24799517])
```

To see a plot, instead of getting a `FittedModel`, use `piecewise_plot()`.  You may also use an existing `FittedModel`.

```
from piecewise import piecewise_plot

# using an existing FittedModel
piecewise_plot(t, v, model=model)

# fitting a model on the fly
piecewise_plot(t, v)
```

<img src="/img/example_regression.png" width="400px">
