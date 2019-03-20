from typing import Sequence, Tuple, Union
from dataclasses import dataclass
from scipy.integrate import odeint

import numpy as np
import logging

from dno.environment import EnvironmentModel

_logger = logging.getLogger(f'windy.model.{__name__}')


@dataclass
class WindyModel:
    v0: float
    pos0: np.array
    alpha: float
    mass: float

    env: EnvironmentModel
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
        self._log.setLevel(logging.getLevelName(self.verbose))
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
            -self.env.aerodynamic_coef.value / self.mass * v[0] * v_mod,
            -self.env.aerodynamic_coef.value / self.mass * v[1] * v_mod - self.env.g,
            -self.env.aerodynamic_coef.value / self.mass * v[2] * v_mod,
        ])
        return dvdt.ravel()

    def position_model(self, state: np.ndarray, time: np.ndarray):
        v = state[:3]
        v_mod = np.linalg.norm(v)
        dvdt = np.array([
            -self.env.aerodynamic_coef.value / self.mass * v[0] * v_mod,
            -self.env.aerodynamic_coef.value / self.mass * v[1] * v_mod - self.env.g,
            -self.env.aerodynamic_coef.value / self.mass * v[2] * v_mod,
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
            -self.env.aerodynamic_coef.value / self.mass * v[0] * v_mod,
            -self.env.aerodynamic_coef.value / self.mass * v[1] * v_mod - self.env.g,
            -self.env.aerodynamic_coef.value / self.mass * v[2] * v_mod,
        ]).ravel()
        drdt = state[:3] + self.env.winds.predict(state[4])
        dalldt = np.hstack((dvdt, drdt))
        return dalldt

    def solve_velocities(self, time: np.ndarray):
        return odeint(self.velocity_model, self.v0, time)

    def solve_positions(self, time: np.ndarray, wind: np.ndarray, v0: np.ndarray, pos0: np.ndarray):
        return odeint(self.position_model, np.hstack((np.asarray(v0) + np.asarray(wind), pos0)), time)

    def solve_positions_alt(self, time: np.ndarray, v0: np.ndarray, pos0: np.ndarray):
        return odeint(self.position_model_alt, np.hstack((np.asarray(v0), pos0)), time)

    def solve(self, target: np.ndarray):
        state = np.hstack((self.v0, self.pos0))
        last_state_stack = np.array((state, ))
        t_start = 0
        while state[4] > target[1]:
            t = np.linspace(t_start, t_start + 1, num=10)
            last_state_stack = self.solve_positions_alt(t, v0=state[:3], pos0=state[3:])
            state = last_state_stack[-1]
            t_start += 1
        starting_point = self.offset_for_target(target, state[3:])
        return starting_point

    @property
    def current_wind(self) -> np.array:
        return self.env.winds.predict(self.pos[1])

    @property
    def aerodynamic_acc(self) -> np.array:
        return -self.v * np.linalg.norm(self.v) * self.env.aerodynamic_coef.value / self.mass

    def step(self, dt: float):
        self._log.debug(f"Current position: {self.pos[1]:.2f}m above ground.")
        self._log.debug(f"Aerodynamic acc: {self.aerodynamic_acc}")
        self._log.debug(f"Winds: {self.current_wind}")
        self.v = np.asarray(self.v) + self.aerodynamic_acc * dt + dt * np.array([0, -self.env.g, 0])
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
