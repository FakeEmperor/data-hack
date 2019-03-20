import argparse


class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('X', type=float, help='X destanation coordinate')
        self.parser.add_argument('Y', type=float, help='Y destanation coordinate')
        self.parser.add_argument('Z', type=float, help='Z destanation coordinate')
        self.parser.add_argument('-H', dest='H0', type=float,
                                 help='High of drop', default=1000)
        self.parser.add_argument('-V', dest='V0', type=float,
                                 help='Started velocity of drop', default=250)
        self.parser.add_argument('-F', dest='F_path', type=str,
                                 help='path to .csv file data with F info', default="data\\F.csv")
        self.parser.add_argument('-W', dest='W_path', type=str,
                                 help='path to .csv file with winds info', default="data\\Wind.csv")
        self.parser.add_argument('-a', dest='alpha', type=float,
                                 help='angle of drop in radians', default=0)
        self.parser.add_argument('-m', dest='m', type=float,
                                 help='mass of object', default=100)
        self.parser.add_argument('-v', dest='v', type=int,
                                 help='verbosity level', default=50)

    def getParser(self):
        return self.parser.parse_args()
