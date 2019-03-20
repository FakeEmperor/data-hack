import logging

import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

from dno.argsparser import ArgsParser
from dno.model import WindyModel
from dno.environment import WindPredictor, EnvironmentModel, AerodynamicCoefficientPredictor


def get_model(parsed):
    wind_table, f_table = get_tables(parsed)
    aero = AerodynamicCoefficientPredictor(f_table)
    winds = WindPredictor(wind_table)
    env = EnvironmentModel(aero, winds)
    pos = np.array([0, parsed.H0, 0])
    return WindyModel(pos0=pos, v0=parsed.V0, alpha=parsed.alpha, mass=parsed.m, env=env, verbose=parsed.v)


def get_tables(parsed):
    return pd.read_csv(parsed.W_path), pd.read_csv(parsed.F_path)


def run(model):
    pos = np.hstack((model.v0, model.pos0))
    t_start = 0
    while pos[4] > 0:
        t = np.linspace(t_start, t_start + 1, num=10)
        y = model.solve_positions_alt(t, v0=pos[:3], pos0=pos[3:])
        pos = y[-1]
        t_start += 1
    return model.offset_for_target((0, 0, 0), pos[3:])


def main():
    parsed = ArgsParser().getParser()
    if not Path(parsed.F_path).is_file():
        raise FileNotFoundError(f"Can't open (Forces) {parsed.F_path} file.")
    if not Path(parsed.W_path).is_file():
        raise FileNotFoundError(f"Can't open (Winds) " + parsed.W_path + " file.")
    model = get_model(parsed)
    print("Started solving...")
    started = datetime.now()
    destination = np.array((parsed.X, parsed.Y, parsed.Z))
    result = model.solve(destination)
    print(f"Result position: {result}, took: {datetime.now() - started}")
    print("Verifying accuracy...")
    print(f"Average error: {model.score(result, destination, dt=0.005)[1]:.2f}m")


if __name__ == "__main__":
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        level=logging.DEBUG,
        datefmt='%I:%M:%S'
    )
    main()
