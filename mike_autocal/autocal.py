import copy
import datetime
import logging
import pickle
import warnings
from pathlib import Path

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler

from mike_autocal.dataio import SimObsPair, SimObsPairCollection
from mike_autocal.measurement_fun import BaseMeasurementFunction
from mike_autocal.mikesimulation import Launcher, RunTimeEvaluation
from mike_autocal.objective_fun import InnerMetric, OuterMetric

warnings.filterwarnings("ignore", category=ExperimentalWarning)

logging.basicConfig()
logger = logging.getLogger("autocal")


class AutoCal:
    def __init__(
        self,
        launcher: Launcher,
        simobs: list[SimObsPair],
        inner_metric: list[InnerMetric],
        outer_metric: list[OuterMetric],
        direction: list[str],
        measurement_functions: list[BaseMeasurementFunction],
        sampler: BaseSampler,
        n_trials: int,
        study_name: str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        evaluation_time: int | str | slice | None = None,
        optuna_log: str | Path = "optuna_journal.log",
        load_if_exists: bool = True,
        verbose: bool = False,
    ):
        self.study_name = study_name
        self.launcher = launcher
        self._simobs = simobs
        self.inner_metric = inner_metric
        self.outer_metric = outer_metric
        self.direction = direction
        self.n_trials = n_trials
        self.measurement_functions = measurement_functions
        self.sampler = sampler
        self.evaluation_time = evaluation_time
        self.optuna_log = Path(optuna_log)
        self.load_if_exists = load_if_exists
        self.base_logdir = Path(launcher.rte.logdir)

        if verbose:
            logger.setLevel(logging.DEBUG)

        self._sanity_check()

    @property
    def simobs(self):
        return SimObsPairCollection(self._simobs)

    def _sanity_check(self):
        if not all(d in ["minimize", "maximize"] for d in self.direction):
            raise ValueError("All elements in 'direction' must be either 'minimize' or 'maximize'")
        if not self.launcher.simfile.exists():
            raise FileNotFoundError(f"{self.launcher.simfile} does not exist")
        if not all(isinstance(mf, BaseMeasurementFunction) for mf in self.measurement_functions):
            raise TypeError("All elements in 'measurement_functions' must be instances of BaseMeasurementFunction")
        if not isinstance(self.sampler, BaseSampler):
            raise TypeError("sampler must be an instance of BaseSampler")
        if not all(isinstance(im, InnerMetric) for im in self.inner_metric):
            raise TypeError("All elements in 'inner_metric' must be instances of InnerMetric")
        if not all(isinstance(om, OuterMetric) for om in self.outer_metric):
            raise TypeError("All elements in 'outer_metric' must be instances of OuterMetric")
        if len(self.inner_metric) != len(self.outer_metric) != len(self.direction):
            raise ValueError("Length of inner_metric, outer_metric and direction must be the same")

    def run(self) -> None:
        """
        Runs the autocal process using the specified parameters.

        The optimization process is run using the specified direction, sampler,
        and number of trials. The optuna results are stored in the specified log file.
        """
        storage = optuna.storages.JournalStorage(
             optuna.storages.journal.JournalFileBackend(self.optuna_log.as_posix()),
        )

        if len(self.direction) == 1:
            direction = self.direction[0]
        else:
            direction = self.direction

        study = optuna.create_study(
            direction=direction,
            sampler=self.sampler,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=self.load_if_exists,
        )

        study.optimize(lambda trial: self._objective(trial), n_trials=self.n_trials)

    def _evaluate_objective(
        self,
        simobs: SimObsPairCollection,
        inner_metric: InnerMetric,
        outer_metric: OuterMetric,
    ) -> float:
        """
        Evaluates the objective function using the inner and outer metrics.

        Args:
            simobs (SimObsPairCollection): The collection of simulation and observation data.
            inner_metric (InnerMetric): The inner metric to evaluate each simulation / observation pair.
            outer_metric (OuterMetric): The outer metric to determine how the inner metrics should be combined.

        Returns:
            float: The evaluation score of the objective function.
        """
        inner_evaluation = inner_metric.evaluate(simobs)
        outer_evaluation = outer_metric.evaluate(inner_evaluation)

        return outer_evaluation

    def _objective(self, trial: optuna.Trial) -> float | tuple[float]:
        """
        Objective function to be optimized by Optuna.

        This function is invoked by Optuna to evaluate the objective function(s)
        for a given set of parameters (measurement functions). For each parameter,
        it suggests a new value, modifies the simulation file accordingly,
        and executes the simulation. The results of the simulation are then updated
        and compared to the observations to evaluate the objective function's performance based
        on the inner and outer metric(s). The function returns a numerical evaluation(s) representing
        how well the parameters perform against the observations.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object that provides
                access to the parameter values to be optimized.

        Returns:
            float | tuple[float]: The evaluation score of the objective function(s), which will be
                minimized or maximized depending on the optimization goal.
        """
        logger.info(f"------------------------------------ Trial: {trial.number} ------------------------------------")

        for measurement_function in self.measurement_functions:
            new_param_value = measurement_function.suggest_new_value(trial)
            self.launcher.simfile = measurement_function.create_new_simfile(
                new_param_value=new_param_value,
                current_simfile=self.launcher.simfile,
                trial_no=trial.number,
                study_name=self.study_name,
            )

        self.launcher.rte.logdir = self.base_logdir / f"_{trial.number}"
        self.launcher.execute_simulation()

        self.simobs.update(
            simulation_results_folder=self.launcher.result_folder,
            time=self.evaluation_time,
        )

        objective_values = []

        for inner_metric, outer_metric in zip(self.inner_metric, self.outer_metric):
            objective_values.append(
                self._evaluate_objective(
                    simobs=self.simobs,
                    inner_metric=inner_metric,
                    outer_metric=outer_metric,
                )
            )

        return tuple(objective_values)
