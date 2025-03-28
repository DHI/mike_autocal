import logging
import multiprocessing
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, TypeVar

import mikeio  # type: ignore
import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mike_autocal.dataio import SimObsPair, SimObsPairCollection
from mike_autocal.objective_fun import InnerMetric, OuterMetric

T = TypeVar("T")

logging.basicConfig()
logger = logging.getLogger("launcher")

np.seterr(divide="ignore")


class RunTimeEvaluation:
    def __init__(
        self,
        simobs: list[SimObsPair] | None = None,
        inner_metric: list[InnerMetric] | InnerMetric | None = None,
        outer_metric: list[OuterMetric] | OuterMetric | None = None,
        frequency: int = 10,
        logdir: str | Path = Path(f"logs/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    ):
        self._simobs = simobs
        self.inner_metric = self._ensure_list(inner_metric, InnerMetric)
        self.outer_metric = self._ensure_list(outer_metric, OuterMetric)
        self.frequency = frequency
        self.logdir = logdir

        self._sanity_check()

    @staticmethod
    def _ensure_list(item: list[T] | T | None, item_type: type[T]) -> list[T]:
        """Ensure the input is always a list, even if a single item is provided."""
        if item is None:
            return []
        if issubclass(type(item), item_type):
            return [item]
        if isinstance(item, list):
            if all(issubclass(type(i), item_type) for i in item):
                return item
            else:
                raise TypeError(f"Expected {item_type} or list[{item_type}], got list{[type(i) for i in item]}")
        raise TypeError(f"Expected {item_type} or list[{item_type}], got {type(item)}")

    @property
    def simobs(self):
        if self._simobs is None:
            return None
        else:
            return SimObsPairCollection(self._simobs)

    def _sanity_check(self):
        provided = [self._simobs is not None, bool(self.inner_metric), bool(self.outer_metric)]

        if any(provided) and not all(provided):
            raise ValueError("If any of simobs, inner_metric, or outer_metric is provided, all three must be provided.")

        if len(self.inner_metric) != len(self.outer_metric):
            raise ValueError("Length of inner_metric and outer_metric must be the same.")

    @property
    def do_runtime_evaluation(self):
        if self._simobs is None:
            return False
        return any(getattr(pair, "pair_type", "point") == "point" for pair in self._simobs)


class Launcher:
    """
    A class to manage the execution of simulation models. This class allows flexibility in running the model
    either on GPU or CPU and handles the environment setup, command building, and process management.

    Attributes:
        simfile (str): Path to the simulation file.
        use_gpu (bool): Flag indicating whether to use GPU. Defaults to True.
        bin_path (str): Path to the simulation binary directory. Defaults to "/teamspace/studios/this_studio/MIKE/2024/bin".
        mikevars_script (str): Path to MIKE environment script. Defaults to "/teamspace/studios/this_studio/MIKE/2024/mikevars.sh".
        mpi_env_script (str): Path to MPI environment script. Defaults to "/teamspace/studios/this_studio/intel/oneapi/mpi/2021.7.0/env/vars.sh".
        allowed_gpu_filetypes (list): List of file extensions that are supported for GPU usage. Defaults to [".m21fm", ".m3fm"].
    """

    bin_path: str = "/teamspace/studios/this_studio/MIKE/2024/bin"
    mikevars_script: str = "/teamspace/studios/this_studio/MIKE/2024/mikevars.sh"
    mpi_env_script: str = "/teamspace/studios/this_studio/intel/oneapi/mpi/2021.7.0/env/vars.sh"
    allowed_gpu_filetypes: list = [".m21fm", ".m3fm"]

    def __init__(
        self,
        simfile: str | Path,
        use_gpu: bool = True,
        n_cores: int | None = None,
        runtimeevaluation: RunTimeEvaluation = RunTimeEvaluation(),
    ):
        """
        Initializes the Launcher with simulation parameters.

        Args:
            simfile (str | Path): Path to the simulation file.
            use_gpu (bool): Flag indicating whether to use GPU (default: True).
            n_cores (int, optional): Number of cores to use for CPU execution or number of GPUs to use for GPU execution (default: None (CPU: max(available CPUs) -1; GPU: 1)).
            runtimeevaluation (RunTimeEvaluation, optional): RunTimeEvaluation object if a runtime evaluation should be written to a tensorboard (default: None).
        """
        self.simfile = Path(simfile)
        self.use_gpu = use_gpu
        self._n_cores = n_cores
        self.rte = runtimeevaluation

    @property
    def logfile(self) -> Path:
        """Returns the log file path with the same name as simfile but `.log` extension."""
        return Path(self.simfile).with_suffix(".log")

    @property
    def file_type(self) -> str:
        """Returns the file type of the simulation file."""
        return Path(self.simfile).suffix

    @property
    def engine(self) -> str:
        """Returns the engine type of the simulation file."""
        return self._determine_engine()

    @property
    def result_folder(self) -> Path:
        return Path(f"{Path(self.simfile)} - Result Files")

    @property
    def n_cpus(self):
        max_cpus = max(multiprocessing.cpu_count(), 1)
        if self._n_cores is None:
            n_cpus = max_cpus - 1
        else:
            if self._n_cores > max_cpus:
                logging.warning(f"Requested {self._n_cores} cores, but only {max_cpus} cores are available. Using {max_cpus} -1 cores instead.")
                n_cpus = max_cpus - 1
            else:
                n_cpus = self._n_cores
        return n_cpus

    @property
    def n_gpus(self):
        max_gpus = self._get_gpu_count()
        if max_gpus == 0:
            return 0
        else:
            if self._n_cores is None:
                n_gpus = 1
            else:
                if self._n_cores > max_gpus:
                    logger.warning(f"Requested {self._n_cores} GPUs, but only {max_gpus} GPUs are available. Using {max_gpus} GPUs instead.")
                    n_gpus = max_gpus
                else:
                    n_gpus = self._n_cores
            return n_gpus

    def __str__(self):
        return f"Launcher(simfile={self.simfile}, use_gpu={self.use_gpu}, n_gpus={self.n_gpus}, n_cpus={self.n_cpus})"

    def _determine_engine(self) -> Optional[str]:
        """
        Determines the appropriate engine based on the simulation file type.

        Returns:
            str: The selected engine type (e.g., "FemEngineSW", "FemEngineHD").

        Raises:
            ValueError: If the simulation file type is unsupported.
        """
        if self.file_type == ".sw":
            return "FemEngineSW"
        elif self.file_type in [".m21fm", ".m3fm"]:
            return "FemEngineHD"
        else:
            raise ValueError(f"Unsupported simulation file type: {self.file_type}.")

    def _read_num_timesteps(self) -> int:
        """
        Reads the number of timesteps from the simulation file.

        Returns:
            int: The number of timesteps for the simulation.

        """
        simfile_pfs = mikeio.read_pfs(self.simfile)

        engine_class = self.engine
        if engine_class not in dir(simfile_pfs):
            raise ValueError(f"Engine class {engine_class} not found in the simulation file")

        time_step_field = getattr(simfile_pfs, engine_class).TIME
        return time_step_field.number_of_time_steps

    def _get_gpu_count(self) -> int:
        """
        Gets the number of available GPUs on the system.

        Returns:
            int: The number of available GPUs.

        """
        return torch.cuda.device_count()

    def _build_cpu_command(self) -> str:
        """
        Builds the command to run the simulation on CPU.

        Returns:
            str: The constructed command to run the simulation on CPU.
        """

        if self.n_cpus > 1:
            logger.info(f"Running simulation on CPU ({self.n_cpus} cores).")
        else:
            logger.info(f"Running simulation on CPU ({self.n_cpus} core).")
        return f"mpirun -n {self.n_cpus} {self.engine} {self.simfile}"

    def _build_gpu_command(self) -> str | None:
        """
        Builds the command to run the simulation on GPU.

        Returns:
            str or None: The command to run the simulation on GPU, or None if no GPU is available.

        If no GPU is detected, will fall back to CPU.
        """

        if self.n_gpus == 0:
            logger.warning("No GPUs detected. Falling back to CPU.")
            return None
        elif self.n_gpus == 1:
            logger.info(f"Running simulation on GPU ({self.n_gpus} GPUs).")
        else:
            logger.info(f"Running simulation on GPU ({self.n_gpus} GPU).")

        return f"mpirun -n {self.n_gpus} {self.engine}GPU {self.simfile}"

    def _run_process(self, command: str) -> None:
        """
        Executes the simulation process using the constructed command.

        Args:
            command (str): The command to execute.

        Logs the process output (stdout and stderr) and tracks progress using `tqdm`.
        """

        writer = SummaryWriter(log_dir=self.rte.logdir)

        full_command = f"bash -c 'source {self.mikevars_script} && source {self.mpi_env_script} && {command}'"
        self.process = subprocess.Popen(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        if self.rte is not None:
            writer.add_scalar("hardware/gpu_count", self._get_gpu_count(), 0)
            writer.add_text("hardware/gpu_used", str(self.use_gpu and self._get_gpu_count() > 0), 0)

        num_timesteps = self._read_num_timesteps()
        timestep_old = 0
        try:
            with tqdm(total=num_timesteps, desc="Processing", unit="step") as pbar:
                start_time = datetime.now()
                if self.process.stdout is not None:
                    for line in self.process.stdout:
                        if "Time step:" in line:
                            try:
                                timestep = int(line.split(":")[1].strip())
                                pbar.update(timestep - timestep_old)
                                timestep_old = timestep
                                if timestep > 0 and (timestep == 1 or timestep % self.rte.frequency == 0):
                                    step_duration = (datetime.now() - start_time).total_seconds()
                                    writer.add_scalar("time/step_duration", step_duration, timestep)
                                    minutes_left = round(
                                        step_duration * (num_timesteps - timestep) / 60,
                                        2,
                                    )
                                    writer.add_scalar("time/minutes_left", minutes_left, timestep)
                                    hours_left = round(
                                        step_duration * (num_timesteps - timestep) / 3600,
                                        2,
                                    )
                                    writer.add_scalar("time/hours_left", hours_left, timestep)

                                    if self.rte.do_runtime_evaluation:
                                        try:
                                            self.rte.simobs.update(self.result_folder)
                                            for inner_metric, outer_metric in zip(
                                                self.rte.inner_metric,
                                                self.rte.outer_metric,
                                            ):
                                                inner_evaluation = inner_metric.evaluate(self.rte.simobs)
                                                outer_evaluation = outer_metric.evaluate(inner_evaluation)
                                                writer.add_scalar(
                                                    f"evaluation/{inner_metric.name}_{outer_metric.name}",
                                                    outer_evaluation,
                                                    timestep,
                                                )
                                        except Exception as e:
                                            logger.warning(f"Failed to evaluate metrics for the runtime evaluation: {e}")

                                start_time = datetime.now()
                            except ValueError as e:
                                logger.error(f"Failed to parse timestep from line: {line.strip()} ({e})")

        except KeyboardInterrupt:
            logger.info("Simulation interrupted. Terminating the process...")
            self.terminate()
            writer.close()
            logger.info("Process terminated successfully.")
            raise

        if self.process.stderr is not None:
            for line in self.process.stderr:
                logger.error(f"{line.strip()}")

        self.process.wait()
        writer.close()

        if self.process.returncode == 0:
            logger.info("Simulation completed successfully!")
        else:
            try:
                with open(self.logfile, "r") as file:
                    lines = file.readlines()
                log_error = "".join(lines[-2:])
                logger.error(f"Simulation failed with exit code {self.process.returncode}\nLast 2 lines of .log file:\n{log_error}")
            except Exception as e:
                logger.error(f"Simulation failed with exit code {self.process.returncode} ({e})")

    def _get_detached_child_processes(self) -> list[psutil.Process]:
        """
        Retrieves a list of detached child processes associated with the simulation.

        This function identifies child processes that were created within a 30-second window
        around the start time of the main simulation process and contain 'FemEngine' in their name.

        Returns:
            list[psutil.Process]: A list of psutil.Process objects representing the detached child processes.
        """

        main_process_start_time = psutil.Process(self.process.pid).create_time()

        time_window_start = main_process_start_time - 30
        time_window_end = main_process_start_time + 30

        return [
            proc
            for proc in psutil.process_iter(["pid", "name", "create_time"])
            if "FemEngine" in proc.info["name"] and time_window_start <= proc.info["create_time"] <= time_window_end
        ]

    def terminate(self):
        """Terminates the simulation process and any detached child processes.

        This function attempts to terminate the main simulation process and any
        detached child processes that were started within a specific time window and contain 'FemEngine'.
        """
        detached_child_processes = self._get_detached_child_processes()

        self.process.terminate()
        self.process.wait()

        for child in detached_child_processes:
            try:
                child.terminate()
                child.wait()
            except psutil.NoSuchProcess:
                pass

    def execute_simulation(self) -> "Launcher":
        """
        Runs the simulation based on the selected configuration (GPU or CPU).

        Determines the proper execution method (GPU or CPU), builds the appropriate command,
        and runs the process.

        If no GPU is available, it will fall back to CPU execution.
        """
        if self.use_gpu and self.file_type not in self.allowed_gpu_filetypes:
            raise TypeError(f"Files of type '{self.file_type}' are not supported for GPU execution. Please disable GPU usage.")

        command = None
        if self.use_gpu:
            command = self._build_gpu_command()

        if not self.use_gpu or command is None:
            command = self._build_cpu_command()
            if self.rte.do_runtime_evaluation and self.n_cpus > 1:
                self.rte = RunTimeEvaluation(frequency=self.rte.frequency, logdir=self.rte.logdir)
                logger.warning("Runtime evaluation is not supported for CPU execution on multiple cores. Proceeding without runtime evaluation.")

        if command:
            self._run_process(command)
        else:
            logger.error("Failed to build a valid command for simulation execution.")

        return self
