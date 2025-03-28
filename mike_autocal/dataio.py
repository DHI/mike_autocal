import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mikeio

logger = logging.getLogger("autocal")


@dataclass
class BaseData(ABC):
    file_path: str | Path
    item: int
    data: mikeio.Dataset = field(init=False)

    def update(self, new_results_folder: str | None = None, time=int | str | slice | None):
        """
        Updates the data attribute by reading the specified item from the file path
        using mikeio. Raises a ValueError if the file cannot be read.

        Args:
            new_results_folder (str): The path to the new results folder.
            time (int | str | slice | None): The time index or slice to read from the file. If None reads all timesteps. Defaults to None.

        Raises:
            ValueError: If there is an issue reading the file at self.file_path.
        """
        if new_results_folder is None:
            pass
        else:
            self.file_path = self._update_file_path(new_results_folder)
        try:
            self.data = mikeio.read(self.file_path, items=self.item, time=time)
        except ValueError as e:
            raise ValueError(f"Failed to read {self.file_path}\n{e}")

    @abstractmethod
    def _update_file_path(self, simulation_results_folder: str | Path) -> Path:
        pass


@dataclass
class ObservationData(BaseData):
    """Represents an observation dataset."""

    def _update_file_path(self) -> Path:
        return Path(self.file_path)


@dataclass
class SimulationData(BaseData):
    """Represents a simulation dataset."""

    def _update_file_path(self, simulation_results_folder: str) -> Path:
        return Path(simulation_results_folder) / Path(self.file_path).name


@dataclass
class SimObsPair:
    name: str
    sim: SimulationData
    obs: ObservationData
    pair_type: Literal["point", "track"] = "point"


@dataclass
class SimObsPairCollection:
    simobs_pairs: list[SimObsPair] = field(default_factory=list)

    def update(self, simulation_results_folder: str, time=int | str | slice | None):
        """
        Updates all the simulation and observation data in the collection by calling
        their respective update methods. This means the data from the files will be reloaded.

        Args:
            simulation_results_folder (str): The path to the simulation results folder.
            time (int | str | slice | None): The time index or slice to read from the file. If None reads all timesteps. Defaults to None.
        """
        for pair in self.simobs_pairs:
            pair.sim.update(simulation_results_folder, time=time)
            pair.obs.update(time=time)
