# abtestwise

A lightweight Python toolkit for **frequentist and Bayesian A/B testing**.

ABTestWise supports both:

* Binary outcomes such as conversion, churn, or success rates
* Continuous outcomes such as revenue, time spent, scores, or average days

The package is built around simple two-group experiments with:

* Control A
* Treatment B
* Frequentist inference
* Bayesian posterior simulation
* Practical decision metrics
* Simple plots

The sign convention is always:

```text
Treatment B - Control A
```

Positive values favor Treatment B. Negative values favor Control A.

## Install

Install from PyPI:

```bash
pip install abtestwise
```

## Development install

To work on the package locally with test dependencies:

```bash
pip install -e ".[dev]"
```

## Binary A/B tests

Use `BinaryABTest` for outcomes with two possible values such as converted/not converted, success/failure, or retained/churned.

ABTestWise provides three binary constructors:

```text
BinaryABTest.from_counts()
BinaryABTest.from_samples()
BinaryABTest.from_dataframe()
```

All three use the same underlying analysis.

### Aggregate counts

Use `from_counts()` when your experiment is already aggregated.

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
```

You can also access individual results:

```python
result.control_rate
result.treatment_rate
result.absolute_lift
result.relative_lift
result.z_statistic
result.p_value
result.confidence_interval_bounds
result.posterior_mean_lift
result.posterior_median_lift
result.prob_treatment_better
result.prob_control_better
result.credible_interval_bounds
result.expected_loss_treatment
result.expected_loss_control
```

### Raw binary samples

Use `from_samples()` when control and treatment observations are already separated.

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

Samples must:

* Be one-dimensional
* Be non-empty
* Contain only finite values
* Contain only binary `0/1` values

Boolean values are also accepted.

### Binary DataFrame input

Use `from_dataframe()` when your experiment is stored in tabular data.

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

`from_dataframe()` does not make pandas a required ABTestWise runtime dependency.

If you pass a pandas DataFrame, pandas must already be installed in your environment. The method also works with DataFrame-like objects that support ordinary column access.

Internally:

```text
from_dataframe()
    -> from_samples()
    -> from_counts()
```

This keeps the statistical implementation consistent across all three constructors.

## Binary frequentist analysis

Binary tests use a two-sided pooled two-proportion z-test.

ABTestWise reports:

```text
Control rate
Treatment rate
Absolute lift
Relative lift
z statistic
p-value
Confidence interval for Treatment - Control
```

The frequentist confidence interval uses the **Newcombe method based on Wilson score intervals** for the difference between two independent proportions.

The default confidence level is:

```python
confidence_level=0.95
```

It can be changed:

```python
test = BinaryABTest.from_counts(
    120,
    1000,
    145,
    1000,
    confidence_level=0.90,
)
```

## Binary Bayesian analysis

Binary tests use a Beta-Binomial model.

For each arm:

```text
Posterior = Beta(
    prior_alpha + successes,
    prior_beta + failures
)
```

The default prior is:

```python
prior_alpha=1.0
prior_beta=1.0
```

which gives:

```text
Beta(1, 1)
```

The prior is configurable.

```python
test = BinaryABTest.from_counts(
    120,
    1000,
    145,
    1000,
    prior_alpha=2.0,
    prior_beta=2.0,
    seed=42,
)
```

The same prior specification is currently applied to both arms.

ABTestWise draws posterior samples for each arm and calculates:

```text
Treatment posterior rate - Control posterior rate
```

The posterior lift distribution is used to calculate:

* Posterior mean lift
* Posterior median lift
* Probability Treatment B is better
* Probability Control A is better
* Equal-tailed credible interval
* Probability lift exceeds a threshold
* Do No Harm probability
* Probability of harm beyond a margin
* Expected loss from choosing Treatment B
* Expected loss from choosing Control A

The default Bayesian credible interval is:

```python
credible_interval=0.95
```

It is configurable independently from the frequentist confidence level.

```python
test = BinaryABTest.from_counts(
    120,
    1000,
    145,
    1000,
    confidence_level=0.90,
    credible_interval=0.95,
    seed=42,
)
```

## Binary practical thresholds

`prob_lift_above()` calculates the posterior probability that lift exceeds a user-defined threshold.

For binary metrics, thresholds use raw proportion units.

```python
result.prob_lift_above(0.01)
```

This calculates:

```text
P(Treatment - Control > 0.01)
```

A threshold of `0.01` means 1 percentage point.

## Binary Do No Harm

`prob_no_harm()` calculates the probability that Treatment B is not worse than Control A by more than a chosen margin.

```python
result.prob_no_harm(0.005)
```

This calculates:

```text
P(lift >= -0.005)
```

A margin of `0.005` means 0.5 percentage points.

The complement is:

```python
result.prob_harm_above(0.005)
```

which calculates:

```text
P(lift < -0.005)
```

For the same margin:

```text
P(No Harm) + P(Harm Above Margin) = 1
```

## Continuous A/B tests

Use `ContinuousABTest` for numeric outcomes such as:

* Revenue
* Time spent
* Scores
* Average days
* Session length
* Other continuous measurements

ABTestWise provides two continuous constructors:

```text
ContinuousABTest.from_samples()
ContinuousABTest.from_dataframe()
```

There is no `from_counts()` constructor for continuous metrics.

### Continuous raw samples

```python
import numpy as np

from abtestwise import ContinuousABTest

control = np.array([10.0, 11.5, 9.5, 12.0, 10.5])
treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5, 13.0])

test = ContinuousABTest.from_samples(
    control=control,
    treatment=treatment,
    seed=42,
)

result = test.run()

print(result.summary())
```

Continuous samples must:

* Be one-dimensional
* Contain numeric values
* Contain only finite values
* Contain at least two observations per arm
* Have non-zero variance

Control and treatment may have different sample sizes.

### Continuous DataFrame input

```python
import pandas as pd

from abtestwise import ContinuousABTest

df = pd.DataFrame(
    {
        "variant": [
            "control",
            "control",
            "control",
            "treatment",
            "treatment",
            "treatment",
        ],
        "revenue": [10.0, 11.5, 9.5, 12.0, 13.5, 11.0],
    }
)

test = ContinuousABTest.from_dataframe(
    df,
    group_col="variant",
    outcome_col="revenue",
    control="control",
    treatment="treatment",
    seed=42,
)

result = test.run()
```

As with binary tests, pandas is optional.

`from_dataframe()` extracts the selected control and treatment observations and delegates to `from_samples()`.

## Continuous frequentist analysis

Continuous tests use a two-sided **Welch t-test**:

```python
scipy.stats.ttest_ind(
    treatment,
    control,
    equal_var=False,
    alternative="two-sided",
)
```

Treatment is passed first so the direction remains:

```text
Treatment B - Control A
```

ABTestWise reports:

* Control sample size
* Treatment sample size
* Control mean
* Treatment mean
* Observed mean difference
* Welch t statistic
* Two-sided p-value
* Welch degrees of freedom
* Confidence interval for the mean difference

The default confidence level is:

```python
confidence_level=0.95
```

It can be changed:

```python
test = ContinuousABTest.from_samples(
    control,
    treatment,
    confidence_level=0.90,
    seed=42,
)
```

## Continuous Bayesian analysis

Continuous tests use SciPy's `mvsdist()` posterior distributions for a Normal population with unknown mean and variance.

ABTestWise uses this as a standard non-informative/reference-prior approach. Users do not provide continuous prior hyperparameters.

For each arm, ABTestWise obtains the posterior distribution for the population mean and draws samples from it.

The posterior difference is:

```text
Treatment population mean - Control population mean
```

The same reference-prior methodology is used each time an analysis is run. ABTestWise does not store or reuse prior values from earlier analyses.

SciPy documents the objective treatment of variance and standard deviation in this model using a Jeffreys prior. This reference-prior approach does not require users to specify numerical prior hyperparameters.

ABTestWise does not run MCMC and does not require PyMC.

The posterior mean-difference distribution is used to calculate:

* Posterior mean difference
* Posterior median difference
* Probability Treatment B is better
* Probability Control A is better
* Equal-tailed credible interval
* Probability the difference exceeds a threshold
* Do No Harm probability
* Probability of harm beyond a margin
* Expected loss from choosing Treatment B
* Expected loss from choosing Control A

The default Bayesian credible interval is:

```python
credible_interval=0.95
```

It can be configured independently:

```python
test = ContinuousABTest.from_samples(
    control,
    treatment,
    confidence_level=0.90,
    credible_interval=0.95,
    seed=42,
)
```

## Continuous practical thresholds

Continuous thresholds use the metric's original units.

For example:

```python
result.prob_difference_above(0.25)
```

If the metric is revenue measured in dollars, this means:

```text
P(Treatment mean - Control mean > $0.25)
```

If the metric is time measured in minutes:

```python
result.prob_difference_above(1.0)
```

means a difference greater than one minute.

## Continuous Do No Harm

The continuous Do No Harm calculation uses the same logic as the binary version, but margins are in the metric's original units.

```python
result.prob_no_harm(0.50)
```

This calculates:

```text
P(mean difference >= -0.50)
```

For a revenue metric measured in dollars, this means Treatment B is not worse than Control A by more than $0.50.

The complement is:

```python
result.prob_harm_above(0.50)
```

## Expected loss

Both binary and continuous results report expected loss for choosing either arm.

For Treatment B:

```text
mean(max(-(Treatment - Control), 0))
```

For Control A:

```text
mean(max(Treatment - Control, 0))
```

For binary metrics, expected loss is a proportion difference and is displayed in percentage points in the summary.

For continuous metrics, expected loss stays in the metric's original units.

## Binary plotting

```python
import matplotlib.pyplot as plt

result.plot_lift_distribution()
result.plot_probability_bar()

plt.show()
```

`plot_lift_distribution()` shows:

* Posterior lift distribution
* No-difference line at zero
* Posterior median
* Credible interval bounds

Binary lift is displayed in percentage points.

`plot_probability_bar()` compares:

```text
P(Treatment B > Control A)
P(Control A > Treatment B)
```

## Continuous plotting

```python
import matplotlib.pyplot as plt

result.plot_difference_distribution()
result.plot_probability_bar()

plt.show()
```

`plot_difference_distribution()` shows:

* Posterior mean-difference distribution
* No-difference line at zero
* Posterior median
* Credible interval bounds

Continuous differences are displayed in the metric's original units.

Plotting methods return Matplotlib `Axes` objects and do not call `plt.show()` automatically.

## Raw results

Both result types provide:

```python
result.to_dict()
```

The returned dictionary contains scalar result values but does not include the full posterior simulation array.

Posterior samples remain available directly on the result object if needed.

Binary:

```python
result.lift_samples
```

Continuous:

```python
result.mean_difference_samples
```

## Groups and sign convention

ABTestWise always uses:

* **Control A** as the baseline group
* **Treatment B** as the test group
* **Treatment B - Control A** as the effect direction

For binary tests:

```text
lift = treatment_rate - control_rate
```

For continuous tests:

```text
mean_difference = treatment_mean - control_mean
```

A positive result means Treatment B is higher than Control A.

ABTestWise currently uses a higher-is-better interpretation when using terms such as "better."

Explicit lower-is-better support is not part of the current API.

## Continuous model assumption

The continuous Bayesian model assumes a Normal population model for the mean and variance.

This simple model may not be a good representation for metrics that are:

* Highly skewed
* Heavy tailed
* Strongly zero inflated

Revenue per user is one example where this may matter.

ABTestWise does not automatically test for these conditions or choose a different distribution. Users should decide whether the Normal model is appropriate for their metric and experiment.

Welch's t-test is generally robust in many practical settings, especially with reasonable sample sizes, but the quality of the Bayesian Normal model still depends on the data-generating process.

## Current scope

ABTestWise is intentionally focused on simple two-arm experimentation.

### Binary

* Aggregate counts
* Raw binary samples
* DataFrame-like input
* Two-sided pooled two-proportion z-test
* Newcombe/Wilson confidence interval
* Configurable Beta-Binomial prior
* Posterior simulation
* Equal-tailed credible interval
* Practical lift thresholds
* Do No Harm probabilities
* Expected loss
* Posterior plots

### Continuous

* Raw continuous samples
* DataFrame-like input
* Unequal sample sizes
* Two-sided Welch t-test
* Welch confidence interval
* SciPy `mvsdist()` reference-prior Bayesian analysis
* Posterior simulation for the mean difference
* Equal-tailed credible interval
* Practical difference thresholds
* Do No Harm probabilities
* Expected loss
* Posterior plots

ABTestWise does not aim to replace Statsmodels, SciPy, PyMC, or a general Bayesian modeling framework.

## Runtime dependencies

ABTestWise keeps its runtime dependencies small:

```text
NumPy
SciPy
Matplotlib
```

pandas is optional and is not required at runtime.

## References

Continuous frequentist analysis uses SciPy's Welch t-test implementation:

```text
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
```

Continuous Bayesian analysis uses SciPy's `mvsdist()`:

```text
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mvsdist.html
```

Related SciPy documentation:

```text
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bayes_mvs.html
```

Additional reference for the Normal mean and variance Bayesian treatment:

```text
https://www.itl.nist.gov/div898/strd/mcmc/mcmc06_cmd.html
```

## Development

Run the full test suite with:

```bash
python -m pytest -q
```
