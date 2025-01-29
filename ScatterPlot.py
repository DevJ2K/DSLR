import matplotlib.pyplot as plt
import pandas as pd
from utils import save_plot

class ScatterPlot:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df.dropna(inplace=True)
        self.df_only_nb = self.df.iloc[:, 6:]

    def render(self):
        nb_col = 13
        nb_row = 13
        plt.style.use('ggplot')
        _, ax = plt.subplots(nb_row, nb_col, tight_layout=True, figsize=(40, 12))
        for i, feature_1 in enumerate(self.df_only_nb):
            for j, feature_2 in enumerate(self.df_only_nb):
                plot = ax[i, j]
                plot.scatter(self.df[feature_1], self.df[feature_2])
        save_plot('scatter_plot')
        plt.show()


def main():
    scatter_plot = ScatterPlot('datasets/dataset_train.csv')
    scatter_plot.render()

if __name__ == '__main__':
    main()
