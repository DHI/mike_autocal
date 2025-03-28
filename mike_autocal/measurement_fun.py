import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mikeio
import numpy as np
import optuna

logger = logging.getLogger("autocal")


class BaseMeasurementFunction(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Each subclass must define a name property."""
        pass

    @abstractmethod
    def create_new_simfile(self, new_param_value: list, current_simfile: str, trial_no: int, study_name: str) -> Path:
        """Creates a new simfile based on new parameter values and returns its path"""
        pass

    @abstractmethod
    def suggest_new_value(self, trial: optuna.Trial) -> Any:
        """
        Suggests a new value for the measurement function's parameter to be used
        in the next simulation. The suggested value is based on the trial number
        and the bounds of the parameter.

        Args:
            trial (optuna.Trial): The Optuna trial object that provides
                access to the parameter values to be optimized.

        Returns:
            Any: The suggested new value for the parameter.
        """
        pass

    def create_new_path(self, current_simfile: Path | str, trial_no: int, study_name: str) -> Path:
        """
        Generates a new file path for the simulation file with the updated trial number.

        If the current simulation file does not have a trial number in its name,
        appends "_trial_X" to the filename, where X is the trial number. If the
        trial number is already present, it updates the trial number to the
        specified new trial number.

        Args:
            current_simfile (Path): The current simulation file path.
            trial_no (int): The trial number to be used in the new file name.
            study_name (str): The name of the study

        Returns:
            Path: The path object with the new file name incorporating the trial number.
        """
        if f"{study_name}_trial_" not in current_simfile.as_posix():
            return Path(current_simfile).with_stem(f"{current_simfile.stem}_{study_name}_trial_{trial_no}")
        else:
            return Path(current_simfile).with_stem(current_simfile.stem.replace(f"_trial_{trial_no - 1}", f"_trial_{trial_no}"))


class ConditioningFile(BaseMeasurementFunction):
    def __init__(self, filename: str, item_name: str, low: float, high: float, step: float | None = None):
        """
        Initialize a Conditioning file instance. Conditioning files can for instance be bed roughness or smagorinsky coefficient files.

        Args:
            filename (str): The path to the base conditioning file.
            item_name (str): The name of the item in the conditioning file to read values from.
            low (float): Lower endpoint of the range of suggested values.
            high (float): Higher endpoint of the range of suggested values.
            step (float, optional): The step size to use when suggesting new values. Defaults to None.
        """

        self.base_filename = Path(filename)
        self.current_filename = Path(filename)
        self.item_name = item_name
        self.low = low
        self.high = high
        self.step = step

        self._zones = self._find_zones()

    def _find_zones(self, path: str | Path | None = None) -> list[np.ndarray]:
        """
        Finds all zones in the current conditioning file that have the same value.
        Each zone is represented as a numpy array of indices where the value
        is the same.

        Args:
            path (str | Path, optional): The path to read the conditioning file from.
                Defaults to None.

        Returns:
            list[np.ndarray]: The list of zones found in the conditioning file.
        """
        zones = []
        for value in np.unique(self.values):
            zones.append(np.where(self.values == value))

        return zones

    def suggest_new_value(self, trial: optuna.Trial) -> list[float]:
        new_values = []
        for i, zone in enumerate(self._zones):
            new_values.append(trial.suggest_float(f"{self.name.replace(' ', '_')}_{i}", self.low, self.high, step=self.step))

        logger.debug(f"New {self.name} file values: {np.round(np.array(new_values), 4)}")

        return new_values

    @property
    def values(self) -> list[float | int]:
        """
        Reads values from the current conditioning file.

        Raises:
            ValueError: If there is an issue reading the file.
        """
        try:
            ds = mikeio.read(self.current_filename, items=self.item_name)
        except ValueError:
            raise ValueError(f"Failed to read {self.current_filename}")

        return ds[self.item_name].values

    def _create_new_conditioning_file(self, new_param_values: list, trial_no: int, study_name: str) -> Path:
        """
        Creates a new conditioning file by modifying the values in the current conditioning file.

        The new file is saved with a filename that is the same as the base file, but with "_trial_X" appended to the filename,
        where X is the trial number.

        Args:
            new_param_values (list): The new values to use for the file.
            trial_no (int): The trial number to use when naming the new conditioning file.
            study_name (str): The name of the study.

        Returns:
            str: The path to the new conditioning file.
        """

        try:
            ds = mikeio.read(self.current_filename, items=self.item_name)
        except ValueError:
            raise ValueError(f"Failed to read {self.current_filename}")

        new_ds = ds.copy()
        for i, zone in enumerate(self._zones):
            new_ds[self.item_name].values[zone] = new_param_values[i]

        new_filename = self.base_filename.with_stem(f"{self.base_filename.stem}_{study_name}_trial_{trial_no}")

        new_ds.to_dfs(new_filename)

        self.current_filename = new_filename

        logger.debug(f"New {self.name} file created: {new_filename}")

        return Path(new_filename)


class BedRoughnessFile(ConditioningFile, BaseMeasurementFunction):
    @property
    def name(self) -> str:
        return "Bed Roughness"

    def create_new_simfile(self, new_param_value: list, current_simfile: str | Path, trial_no: int, study_name: str) -> Path:
        new_conditioning_file = self._create_new_conditioning_file(new_param_value, trial_no, study_name)

        pfs = mikeio.read_pfs(current_simfile)

        try:
            pfs.HD.BED_RESISTANCE.ROUGHNESS.file_name = f"|{new_conditioning_file}|"

            pfs.HD.BED_RESISTANCE.ROUGHNESS.item_number = 1
            pfs.HD.BED_RESISTANCE.ROUGHNESS.item_name = self.item_name
        except AttributeError as e:
            raise AttributeError(f"Failed to write {self.name} file path to simulation file: {e}")

        new_simfile = self.create_new_path(current_simfile, trial_no, study_name)

        try:
            pfs.write(new_simfile)
            logger.debug(f"New simulation file created: {new_simfile}")
        except Exception as e:
            raise Exception(f"Failed to write new simulation file: {e}")

        return new_simfile


class ManningFile(ConditioningFile, BaseMeasurementFunction):
    @property
    def name(self) -> str:
        return "Manning"

    def create_new_simfile(self, new_param_value: list, current_simfile: str, trial_no: int, study_name: str) -> Path:
        new_conditioning_file = self._create_new_conditioning_file(new_param_value, trial_no, study_name)

        pfs = mikeio.read_pfs(current_simfile)

        try:
            pfs.HD.BED_RESISTANCE.MANNING_NUMBER.file_name = f"|{new_conditioning_file}|"

            pfs.HD.BED_RESISTANCE.MANNING_NUMBER.item_number = 1
            pfs.HD.BED_RESISTANCE.MANNING_NUMBER.item_name = self.item_name
        except AttributeError as e:
            raise AttributeError(f"Error updating {self.name} file: {e}")

        new_simfile = self.create_new_path(current_simfile, trial_no, study_name)

        try:
            pfs.write(new_simfile)
            logger.debug(f"New simulation file created: {new_simfile}")
        except Exception as e:
            raise Exception(f"Failed to write new simulation file: {e}")

        return new_simfile


class SmagorinskyFile(ConditioningFile, BaseMeasurementFunction):
    @property
    def name(self) -> str:
        return "Smagorinsky"

    def create_new_simfile(self, new_param_value: list, current_simfile: str, trial_no: int, study_name: str) -> Path:
        new_conditioning_file = self._create_new_conditioning_file(new_param_value, trial_no, study_name)

        pfs = mikeio.read_pfs(current_simfile)

        try:
            pfs.HD.EDDY_VISCOSITY.HORIZONTAL_EDDY_VISCOSITY.SMAGORINSKY_FORMULATION.file_name = f"|{new_conditioning_file}|"

            pfs.HD.EDDY_VISCOSITY.HORIZONTAL_EDDY_VISCOSITY.SMAGORINSKY_FORMULATION.item_number = 1
            pfs.HD.EDDY_VISCOSITY.HORIZONTAL_EDDY_VISCOSITY.SMAGORINSKY_FORMULATION.item_name = self.item_name
        except AttributeError as e:
            raise AttributeError(f"Error updating {self.name} file: {e}")

        new_simfile = self.create_new_path(current_simfile, trial_no, study_name)

        try:
            pfs.write(new_simfile)
            logger.debug(f"New simulation file created: {new_simfile}")
        except Exception as e:
            raise Exception(f"Failed to write new simulation file: {e}")

        return new_simfile
