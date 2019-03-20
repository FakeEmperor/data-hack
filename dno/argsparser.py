import argparse


class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="This solver can solve numerically, using differential equations,"
                        "a high altitude precision drop. "
                        "It takes into account mass, different winds, "
                        "initial object's velocity and aerodynamic forces to provide accurate results."
                        "Results are validated using high precision simulation.")
        self.parser.add_argument('X', type=float, help='X destination coordinate')
        self.parser.add_argument('Y', type=float, help='Y destination coordinate')
        self.parser.add_argument('Z', type=float, help='Z destination coordinate')
        self.parser.add_argument('-H', dest='H0', type=float,
                                 help='Drop altitude, m', default=1000)
        self.parser.add_argument('-V', dest='V0', type=float,
                                 help='Initial velocity in XZ space, m/s', default=250)
        self.parser.add_argument('-F', dest='F_path', type=str,
                                 help='Path to a CSV file data with F info', default="data\\F.csv")
        self.parser.add_argument('-W', dest='W_path', type=str,
                                 help='path to a CSV file with winds info', default="data\\Wind.csv")
        self.parser.add_argument('-a', dest='alpha', type=float,
                                 help='Velocity angle, grads', default=0)
        self.parser.add_argument('-m', dest='m', type=float,
                                 help='Object\'s mass, kg', default=100)
        self.parser.add_argument('-v', dest='v', type=int,
                                 help='Verbosity level for model simulations', default=50)

    def getParser(self):
        return self.parser.parse_args()
