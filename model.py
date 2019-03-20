from typing import Sequence, Tuple, Union
from dataclasses import dataclass
from scipy.integrate import odeint

import numpy as np
import pandas as pd

import logging

Ca = 0.27340445  # coef before V**2 in Fa
g = 9.81
g_vector = np.array([0, -g, 0])


class WindPredictor:

    def __init__(self, winds, step=100):
        self.winds = winds
        self.step = step

    def predict_wind(self, h):
        wind = self.winds.iloc[int(h // self.step)]
        return np.array([wind['Wx'], 0, wind['Wz']])


class Model:

    def __init__(self, H_0, x_0, z_0, wind_pred, v_0=250, alpha=0.0, m=100.0, dt=0.01):
        self.reset(H_0, x_0, z_0, wind_pred, v_0, alpha, m, dt)

    def reset(self, H_0, x_0, z_0, wind_pred, v_0=250, alpha=0, m=100.0, dt=0.01):
        self.H_0 = H_0
        self.x_0 = x_0
        self.z_0 = z_0
        self.r = np.array([x_0, H_0, z_0])
        self.v = v_0 * np.array([np.cos(alpha), 0, np.sin(alpha)])
        self.a = np.zeros(3)
        self.dt = dt
        self.t = 0
        self.wind_pred = wind_pred
        self.m = m
        self.k = 1 / self.v[0] + 1

    def step(self):
        self.t += self.dt
        h = self.r[1]
        w = self.wind_pred.predict_wind(h)
        self.r = self.r + self.dt * (self.v + w)
        self.v = self.v + self.dt * (self.a + g_vector)
        self.a = -self.v * Ca * np.abs(self.v) / self.m

    def stop_condition(self):
        return self.r[1] <= 0

    def print_state(self):
        vx_predicted = self.m / (- Ca * self.t + self.k * self.m - self.m)
        print(f"time={self.t} r={self.r} v={self.v}, vx_predicted={vx_predicted}")

    def simulate(self):
        while not self.stop_condition():
            self.print_state()
            self.step()
        self.print_state()

    def get_start_point(self):
        result = -self.r
        result[0] += 2*self.x_0
        result[1] = self.H_0
        result[2] += 2*self.z_0
        return result


if __name__ == "__main__":
    winds_df = pd.read_csv("data/Wind.csv")
    winds = winds_df[['Wx', 'Wz']]
    wind_pred = WindPredictor(winds)
    model = Model(H_0=1000, x_0=0, z_0=0, wind_pred=wind_pred, alpha=0)
    model.simulate()
    print(f"Start point: {model.get_start_point()}")

############################
@dataclass
class WindPredictor:
    winds: pd.DataFrame
    _step: int = None
    _logger = logging.getLogger("data_hack.preliminary")

    def __post_init__(self):
        if not self.winds.shape[0]:
            raise ValueError("Wind predictor takes a DataFrame for wind speeds which has at least one point")
        self._logger.info("Trying to detect which type of wind model is needed (calculate deltas in a dataset)...")
        if self.winds.shape[0] == 1:
            self._logger.info("Wind dataset has only one row, we don't need any dynamic step calculation")
            self.predict_dynamic = False
            self._step = 1
        else:
            wind_step_deltas: np.ndarray = np.array(self.winds['Y'])[:-1] - np.array(self.winds['Y'])[1:]
            self._logger.info(wind_step_deltas)
            self.predict_dynamic = not np.all(wind_step_deltas == wind_step_deltas[0])
            self._logger.info(f"Wind dataset will be predicted using dynamic step variation: {self.predict_dynamic}")
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
class WindyModel:
    v0: float
    pos0: np.array
    alpha: float
    mass: float
    aerodynamic_coef: float
    winds: WindPredictor
    g: float = 9.81
    verbose: Union[bool, int] = False

    v: np.ndarray = None
    run_time: float = None
    pos: np.ndarray = None

    @dataclass
    class ModelState:
        v: np.array
        pos: np.array
        time: float

    def __post_init__(self):
        self._log = logging.getLogger(f'{__name__}.{self.__class__.__name__}[{id(self)}]')
        if isinstance(self.verbose, bool):
            self.verbose = logging.DEBUG if self.verbose else logging.CRITICAL
        self._log.setLevel(self.verbose)
        self.alpha = np.deg2rad(self.alpha)
        self.v0 = np.array([self.v0 * np.cos(self.alpha), 0, self.v0 * np.sin(self.alpha)])
        self.reset()

    def reset(self):
        self.v = self.v0
        self.pos = self.pos0
        self.run_time = 0

    # function that returns dy/dt
    def velocity_model(self, v: np.ndarray, time: np.ndarray):
        v_mod = np.linalg.norm(v)
        dvdt = np.array([
            -self.aerodynamic_coef / self.mass * v[0] * v_mod,
            -self.aerodynamic_coef / self.mass * v[1] * v_mod - self.g,
            -self.aerodynamic_coef / self.mass * v[2] * v_mod,
        ])
        return dydt.ravel()

    def position_model(self, state: np.ndarray, time: np.ndarray):
        v = state[:3]
        v_mod = np.linalg.norm(v)
        dvdt = np.array([
            -self.aerodynamic_coef / self.mass * v[0] * v_mod,
            -self.aerodynamic_coef / self.mass * v[1] * v_mod - self.g,
            -self.aerodynamic_coef / self.mass * v[2] * v_mod,
        ]).ravel()
        print(dvdt)
        drdt = state[:3]
        print(drdt)
        dalldt = np.hstack((dvdt, drdt))
        return dalldt

    def position_model_alt(self, state: np.ndarray, time: np.ndarray):
        v = state[:3]
        v_mod = np.linalg.norm(v)
        dvdt = np.array([
            -self.aerodynamic_coef / self.mass * v[0] * v_mod,
            -self.aerodynamic_coef / self.mass * v[1] * v_mod - self.g,
            -self.aerodynamic_coef / self.mass * v[2] * v_mod,
        ]).ravel()
        drdt = state[:3] + self.winds.predict(state[4])
        dalldt = np.hstack((dvdt, drdt))
        return dalldt

    def solve_velocities(self, time: np.ndarray):
        return odeint(self.velocity_model, self.v0, time)

    def solve_positions(self, time: np.ndarray, wind: np.ndarray, v0: np.ndarray, pos0: np.ndarray):
        return odeint(self.position_model, np.hstack((np.asarray(v0) + np.asarray(wind), pos0)), time)

    def solve_positions_alt(self, time: np.ndarray, v0: np.ndarray, pos0: np.ndarray):
        return odeint(self.position_model_alt, np.hstack((np.asarray(v0), pos0)), time)

    @property
    def current_wind(self) -> np.array:
        return self.winds.predict(self.pos[1])

    @property
    def aerodynamic_acc(self) -> np.array:
        return -self.v * np.linalg.norm(self.v) * self.aerodynamic_coef / self.mass

    def step(self, dt: float):
        self._log.debug(f"Current position: {self.pos[1]:.2f}m above ground.")
        self._log.debug(f"Aerodynamic acc: {self.aerodynamic_acc}")
        self._log.debug(f"Winds: {self.current_wind}")
        self.v = np.asarray(self.v) + self.aerodynamic_acc * dt + dt * np.array([0, -self.g, 0])
        self._log.debug(f"Object's speed is: {self.v}")
        v_eff = self.current_wind + self.v
        self._log.debug(f"Effective speed is: {v_eff}")
        self.pos = self.pos + np.asarray(v_eff) * dt
        self._log.debug(f"Position is: {self.pos}")
        self.run_time += dt

    def offset_for_target(self, target: np.ndarray, position: np.ndarray) -> np.ndarray:
        drop_point = target - position + 2 * np.array(self.pos0)
        drop_point[1] = self.pos0[1]
        return drop_point

    def predict(self, target: np.ndarray, reset: bool = True, **run_kwargs) -> np.ndarray:
        if reset:
            self.reset()
        self.run(**run_kwargs)
        starting_point = self.offset_for_target(target, self.pos)
        return starting_point

    def score(self, starting_point: np.array, destination_point: np.array, reset: bool = True, **run_kwargs) -> Tuple[
        np.ndarray, float]:
        saved_pos0 = self.pos0
        try:
            self.pos0 = starting_point
            if reset:
                self.reset()
            self.run(**run_kwargs)
            landing_point = self.state.pos
            landing_point[1] = destination_point[1]
            return landing_point, np.linalg.norm(landing_point - destination_point)
        except Exception as e:
            self.pos0 = saved_pos0
            raise e
        finally:
            self.pos0 = saved_pos0

    @property
    def state(self) -> 'ModelState':
        return self.ModelState(v=self.v, pos=self.pos, time=self.run_time)

    def run(self, max_steps: int = None, dt: float = 0.01) -> Sequence['ModelState']:
        step = 0
        states = []
        while self.pos[1] > 0 and (max_steps is None or step < max_steps):
            self.step(dt)
            step += 1
            states.append(self.state)
            self._log.debug(f"State is: {self.state}")
        self._log.info(f"Final state is: {self.state}")
        if max_steps is not None and step >= max_steps:
            self._log.warning(f"WARN: stopped after maximum iterations: {step}")
        return states