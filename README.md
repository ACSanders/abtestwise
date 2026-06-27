# abtestwise

A lightweight Python toolkit for **binary A/B experiment analysis** using
aggregate count data. Version 0.1 combines frequentist and Bayesian summaries
for binary proportions.

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

```python
from abtestwise import BinaryABTest

test = BinaryABTest.from_counts(
    control_successes=120,
    control_total=1000,
    treatment_successes=145,
    treatment_total=1000,
    prior_alpha=1,
    prior_beta=1,
    n_simulations=100_000,
    credible_interval=0.95,
    seed=42,
)

result = test.run()

print(result.summary())
print(result.prob_lift_above(0.01))
```

`prob_lift_above(0.01)` gives the posterior probability that Treatment B improves
the metric by more than 1 percentage point.

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

- Binary proportions only.
- Aggregate counts only.
- Two groups only.
- Frequentist: two-sided pooled two-proportion z-test.
- Bayesian: beta-binomial posterior simulation with default prior `Beta(1, 1)`.
- Equal-tailed credible interval.
- Expected loss.
- Practical lift thresholds.
- Do-no-harm probabilities using a user-defined harm margin.
- Simple plots.

## Development

Run tests with:

```bash
python -m pytest -q
```
