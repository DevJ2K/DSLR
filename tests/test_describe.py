import sys
import os
import filecmp
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data import Data

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(ROOT_PATH, "tests")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def test_compare_describe_on_dataset_test():
    own_output_path = os.path.join(TEST_PATH, "out/own_describe_with_dataset_test.txt")
    pandas_output_path = os.path.join(TEST_PATH, "out/pandas_describe_with_dataset_test.txt")

    data = Data(os.path.join(ROOT_PATH, "datasets/dataset_test.csv"))
    data.describe(own_output_path)

    with open(pandas_output_path, "w") as fd:

        print(data.df_only_nb.describe(include='all'), file=fd)

    assert filecmp.cmp(own_output_path, pandas_output_path)


def test_compare_describe_on_dataset_train():
    own_output_path = os.path.join(TEST_PATH, "out/own_describe_with_dataset_train.txt")
    pandas_output_path = os.path.join(TEST_PATH, "out/pandas_describe_with_dataset_train.txt")

    data = Data(os.path.join(ROOT_PATH, "datasets/dataset_train.csv"))
    data.describe(own_output_path)

    with open(pandas_output_path, "w") as fd:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)

        print(data.df_only_nb.describe(include='all'), file=fd)

    assert filecmp.cmp(own_output_path, pandas_output_path)


if __name__ == "__main__":
    # print(ROOT_PATH)
    test_compare_describe_on_dataset_test()
