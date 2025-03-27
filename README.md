# MIKE AutoCal

A Python library for automatic calibration of hydrodynamic application with MIKE software. This package provides tools and utilities to streamline the calibration process of MIKE models using modern optimization techniques.

## Features

- Automated calibration of MIKE simulation parameters
- Integration with [Optuna](https://optuna.org/) for efficient parameter optimization
- Support for multiple metrics and measurement functions
- Flexible evaluation time handling
- Progress tracking and logging capabilities

## Installation

Requires Python 3.10 or higher.

```bash
pip install -e .
```

## Basic Usage

```python
from mike_autocal import AutoCal, Launcher, SimObsPair
from mike_autocal.objective_fun import InnerMetric, OuterMetric
from mike_autocal.measurement_fun import BaseMeasurementFunction

# Configure your MIKE simulation launcher
launcher = Launcher(...)

# Define simulation-observation pairs
simobs = [SimObsPair(...)]

# Set up metrics and measurement functions
inner_metrics = [InnerMetric(...)]
outer_metrics = [OuterMetric(...)]
measurement_functions = [BaseMeasurementFunction(...)]

# Initialize the calibration
calibration = AutoCal(
    launcher=launcher,
    simobs=simobs,
    inner_metric=inner_metrics,
    outer_metric=outer_metrics,
    measurement_functions=measurement_functions,
    sampler=your_optuna_sampler,
    n_trials=100,
    study_name="my_calibration"
)

# Run the calibration
calibration.run()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Faro Schäfer (fsch@dhigroup.com)
- Clemens Cremer (clcr@dhigroup.com)
- Jesper Sandvig Mariegaard (jem@dhigroup.com)
- Henrik Andersson (jan@dhigroup.com)

## Development Status

This package is currently in pre-alpha stage. APIs may change without notice.
