import argparse

class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('X', type=float, help='X destanation coordinate')
        self.parser.add_argument('Z', type=float, help='Z destanation coordinate')
        self.parser.add_argument('-H', dest='H0', type=float, help='High of drop. Default = 1000', default=1000)
        self.parser.add_argument('-V', dest='V0', type=float, help='Started velocity of drop. Default = 250', default=250)
        self.parser.add_argument('-F', dest='F_path', type=str, help='path to .csv file data with F info. Default = \"data\\F.csv\"', default="data\\F.csv")
        self.parser.add_argument('-W', dest='W_path', type=str, help='path to .csv file with winds info. Default = \"data\\Wind.csv\"', default="data\\Wind.csv")
        self.parser.add_argument('-a', dest='alpha', type=float, help='angle of drop in radians. Default = 0', default=0)
        self.parser.add_argument('-m', dest='m', type=float, help='mass of object. Default = 1', default=1)

    def getParser(self):
        return self.parser.parse_args()