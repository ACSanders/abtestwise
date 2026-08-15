# abtestwise

A lightweight Python toolkit for **frequentist and Bayesian binary A/B testing**.

ABTestWise accepts aggregate counts, raw binary samples, or DataFrame-like
experiment data while providing the same statistical analysis through a simple,
consistent API.

## Install

Install from PyPI:

```bash
pip install abtestwise
```

## Development install

To work on the package locally (with the test dependencies):

```bash
pip install -e ".[dev]"
```

## Quickstart

### Aggregate counts

Use `from_counts()` when your experiment data is already aggregated:

```python
from abtestwise import BinaryABTest

test = BinaryABTest.from_counts(
    control_successes=120,
    control_total=1000,
    treatment_successes=145,
    treatment_total=1000,
    seed=42,
)

result = test.run()

print(result.summary())
print(result.prob_lift_above(0.01))
```

### Raw samples

Use `from_samples()` when you already have separate control and treatment
observations:

```python
import numpy as np

from abtestwise import BinaryABTest

control = np.array([0, 1, 0, 1, 0, 0, 1])
treatment = np.array([1, 1, 0, 1, 0, 1, 1])

test = BinaryABTest.from_samples(
    control=control,
    treatment=treatment,
    seed=42,
)

result = test.run()
```

Samples must contain binary `0/1` values. Boolean values are also accepted.

### DataFrame input

Use `from_dataframe()` when your experiment is stored in a tabular dataset:

```python
import pandas as pd

from abtestwise import BinaryABTest

df = pd.DataFrame(
    {
        "variant": ["control", "control", "treatment", "treatment"],
        "converted": [0, 1, 1, 1],
    }
)

test = BinaryABTest.from_dataframe(
    df,
    group_col="variant",
    outcome_col="converted",
    control="control",
    treatment="treatment",
    seed=42,
)

result = test.run()
```

`from_dataframe()` does not require pandas as an ABTestWise runtime dependency.
If you pass a pandas DataFrame, pandas must be installed in your environment.
The method works with DataFrame-like objects that support column access.

All three constructors reduce to the same underlying binary A/B test, so
equivalent data produces equivalent statistical results.

`prob_lift_above(0.01)` gives the posterior probability that Treatment B
improves the metric by more than 1 percentage point.

### Do-no-harm checks

`prob_no_harm(margin)` gives the posterior probability that Treatment B is **not**
worse than Control A by more than `margin` (in raw decimal units, so `0.005` means
0.5 percentage points). `prob_harm_above(margin)` is its complement.

```python
result.prob_no_harm(0.005)     # P(lift >= -0.005): B is not worse by more than 0.5pp
result.prob_harm_above(0.005)  # P(lift <  -0.005): B is worse by more than 0.5pp
```

Raw result values are also available:

```python
result.to_dict()
```

## Plotting

```python
import matplotlib.pyplot as plt

result.plot_lift_distribution()
result.plot_probability_bar()

plt.show()
```

The lift distribution plot shows posterior lift in percentage points.

The probability bar plot shows:

```text
P(Treatment B > Control A)
P(Control A > Treatment B)
```

## Groups and sign convention

In product A/B testing terms:

- **Control (A)** is the baseline group.
- **Treatment (B)** is the test group or variant B.
- **Lift is always Treatment B - Control A.**
- **Positive lift means Treatment B is better than Control A.**
- **Negative lift means Control A is better than Treatment B.**

## Scope

Current package scope:

- Binary proportions.
- Aggregate counts, raw binary samples, and DataFrame-like experiment data.
- Two-group comparisons.
- Frequentist: two-sided pooled two-proportion z-test.
- Bayesian: Beta-Binomial posterior simulation with default prior `Beta(1, 1)`.
- Equal-tailed credible intervals.
- Expected loss.
- Practical lift thresholds.
- Do-no-harm probabilities using a user-defined harm margin.
- Simple plots.

## Development

Run tests with:

```bash
python -m pytest -q
```
