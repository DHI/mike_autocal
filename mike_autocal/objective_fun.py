import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import modelskill as ms
import numpy as np
import pandas as pd
from scipy.stats import hmean

from mike_autocal.dataio import SimObsPairCollection
from mike_autocal.utils import get_memory_usage

logger = logging.getLogger("autocal")


@dataclass
class InnerEvaluation:
    """The output of the InnerMetrics"""

    metric: str
    # pair_type: list[str]
    names: list[str]
    values: list[float]
    n: list[int]

    def __post_init__(self):
        if not (len(self.names) == len(self.values) == len(self.n)):
            raise ValueError(f"Length mismatch: names({len(self.names)}), values({len(self.values)}), n_points({len(self.n)})")

    def __str__(self):
        df_str = self.to_dataframe().to_string()
        return f"InnerEvaluation(metric={self.metric}):\n{df_str}"

    def to_dataframe(self):
        df = pd.DataFrame(
            {
                "values": self.values,
                "n": self.n,
                # "pair_type": self.pair_type
            },
            index=self.names,
        )
        df.index.name = "name"

        return df


class InnerMetric(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def evaluate(self, simobs: SimObsPairCollection):
        pass

    def _match_track_pairs(self, simobs: SimObsPairCollection, cc: ms.ComparerCollection | None) -> ms.ComparerCollection:
        track_pairs = [pair for pair in simobs.simobs_pairs if pair.pair_type == "track"]

        for pair in track_pairs:
            gc.collect()
            obs = ms.TrackObservation(data=pair.obs.data, name=pair.name)
            sim = ms.model_result(data=pair.sim.data, name=pair.name)
            matched = ms.match(obs, sim)
            cc = matched if cc is None else cc + matched

        return cc

    def _match_point_pairs(self, simobs: SimObsPairCollection, cc: ms.ComparerCollection | None) -> ms.ComparerCollection:
        point_pairs = [pair for pair in simobs.simobs_pairs if pair.pair_type == "point"]

        for pair in point_pairs:
            gc.collect()
            sim = ms.PointModelResult(pair.sim.data, name=pair.name)
            obs = ms.PointObservation(data=pair.obs.data, name=pair.name)
            matched = ms.match(obs, sim)
            cc = matched if cc is None else cc + matched

        return cc


class RMSEInnerMetric(InnerMetric):
    @property
    def name(self):
        return "RMSE"

    def evaluate(self, simobs: SimObsPairCollection):
        logger.debug(f"Memory usage: {get_memory_usage():.2f} MB")

        cc = None
        cc = self._match_point_pairs(simobs=simobs, cc=cc)
        cc = self._match_track_pairs(simobs=simobs, cc=cc)

        comparison = cc.skill().reset_index()
        inner_evaluation = InnerEvaluation(
            metric=self.name, names=list(comparison["model"]), values=list(comparison["rmse"]), n=list(comparison["n"])
        )

        logger.debug(f"Memory usage: {get_memory_usage():.2f} MB")
        logger.info(inner_evaluation)

        return inner_evaluation


class CCInnerMetric(InnerMetric):
    @property
    def name(self):
        return "CC"

    def evaluate(self, simobs: SimObsPairCollection):
        logger.debug(f"Memory usage: {get_memory_usage():.2f} MB")

        cc = None
        cc = self._match_point_pairs(simobs=simobs, cc=cc)
        cc = self._match_track_pairs(simobs=simobs, cc=cc)

        comparison = cc.skill().reset_index()
        inner_evaluation = InnerEvaluation(metric=self.name, names=list(comparison["model"]), values=list(comparison["cc"]), n=list(comparison["n"]))

        logger.debug(f"Memory usage: {get_memory_usage():.2f} MB")
        logger.info(inner_evaluation)

        return inner_evaluation


class OuterMetric(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def evaluate(self, inner_evaluation: InnerEvaluation):
        pass


class HMEANOuterMetric(OuterMetric):
    @property
    def name(self):
        return "Harmonic Mean"

    def evaluate(self, inner_evaluation: InnerEvaluation):
        return hmean(inner_evaluation.values)


class AMEANOuterMetric(OuterMetric):
    @property
    def name(self):
        return "Arithmetic Mean"

    def evaluate(self, inner_evaluation: InnerEvaluation):
        return np.mean(inner_evaluation.values)
