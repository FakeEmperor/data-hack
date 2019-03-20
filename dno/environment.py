import logging
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


_logger = logging.getLogger(__name__)


@dataclass
class AerodynamicCoefficientPredictor:
    Fa: pd.DataFrame
    split: float = 0.1

    _train: pd.DataFrame = None
    _test: pd.DataFrame = None

    def __post_init__(self):
        self.Fa["V^2"] = self.Fa['V'] ** 2
        self._train, self._test = train_test_split(self.Fa, test_size=0.1)
        self.lr = LinearRegression(normalize=False)
        self._value = None

    def score(self) -> float:
        return self.lr.score(self._test["V^2"].values.reshape(-1, 1), self._test["Fa"])

    @property
    def value(self) -> float:
        """
        Will train a linear regression upon the dataset and look for coefficients.
        :return:
        """
        if self._value is None:
            self.lr.fit(self._train["V^2"].values.reshape(-1, 1), self._train["Fa"])
            self._value = self.lr.coef_[0]
        return self._value


@dataclass
class WindPredictor:
    winds: pd.DataFrame
    _step: int = None

    def __post_init__(self):
        if not self.winds.shape[0]:
            raise ValueError("Wind predictor takes a DataFrame for wind speeds which has at least one point")
        _logger.info("Trying to detect which type of wind model is needed (calculate deltas in a dataset)...")
        if self.winds.shape[0] == 1:
            _logger.info("Wind dataset has only one row, we don't need any dynamic step calculation")
            self.predict_dynamic = False
            self._step = 1
        else:
            wind_step_deltas: np.ndarray = np.array(self.winds['Y'])[:-1] - np.array(self.winds['Y'])[1:]
            _logger.info(wind_step_deltas)
            self.predict_dynamic = not np.all(wind_step_deltas == wind_step_deltas[0])
            _logger.info(f"Wind dataset will be predicted using dynamic step variation: {self.predict_dynamic}")
            if not self.predict_dynamic:
                self._step = np.abs(wind_step_deltas[0].ravel())

    def predict(self, h: float):
        if self.predict_dynamic:
            return self._predict_dynamic_step(h)
        else:
            return self._predict_static_step(h)

    def _predict_static_step(self, h: float):
        wind = self.winds.iloc[min(int(h // self._step), self.winds.shape[0] - 1)]
        return np.array([wind['Wx'], 0, wind['Wz']])

    def _predict_dynamic_step(self, h: float):
        deltas = self.winds['Y'] - h
        wind = self.winds.iloc[deltas[deltas < 0].idxmax()]
        return np.array([wind['Wx'], 0, wind['Wz']])


@dataclass
class EnvironmentModel:
    aerodynamic_coef: AerodynamicCoefficientPredictor
    winds: WindPredictor
    g: float = 9.81
