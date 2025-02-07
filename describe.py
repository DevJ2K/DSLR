#!/bin/python3

from Colors import *
from utils import print_error
from Data import Data, DataError
import csv
import numpy
import os
import sys


def main(file: str = None) -> None:
    if len(sys.argv) != 2 and file is None:
        print(f"{BHWHITE}Usage: ./{os.path.basename(__file__)} <file>")
        return
    if len(sys.argv) == 2:
        data = Data(file=sys.argv[1])
    else:
        data = Data(file=file)
    data.describe(FULL_FIELD=True)

if __name__ == "__main__":
    try:
        main("datasets/dataset_test.csv")
    except Exception as e:
        print_error(e)
