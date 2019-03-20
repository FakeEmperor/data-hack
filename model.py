import numpy as np
import pandas as pd
import math

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
        self.r = self.r + self.dt * (self.v + w )
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
