from model import WindyModel, WindPredictor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from argsparser import ArgsParser
from pathlib import Path

def getLrCoef(F_table):
    # TRAIN LINEAR REGRESSION TO FIND AERODYNAMIC COEFFICIENT
    F_table["V^2"] = F_table['V'] ** 2
    train, test = train_test_split(F_table, test_size=0.1)
    lr = LinearRegression(normalize=False)
    lr.fit(train["V^2"].values.reshape(-1, 1), train["Fa"])
    return lr.coef_[0]

def getModel(parser):
    Wind_table, F_table = getWindFTables(parser)
    winds = WindPredictor(Wind_table, F_table)
    pos = np.array([parser.X, parser.H0, parser.Y])
    return WindyModel(winds=winds, pos0=pos, v0=parser.V0, alpha=parser.alpha, mass=parser.m, aerodynamic_coef=getLrCoef(F_table))

def getWindFTables(parser):
    F_table = pd.read_csv(parser.W_path)
    Wind_table = pd.read_csv(parser.F_path)
    return F_table, Wind_table

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
    parser = ArgsParser().getParser()
    if not Path(parser.F_path).is_file():
        print("Can't open " + parser.F_path + " file.")
        return
    if not Path(parser.W_path).is_file():
        print("Can't open " + parser.W_path + " file.")
        return
    model = getModel(parser)
    result = run(model)
    print(result)
    return

if __name__ == "__main__":
    main()