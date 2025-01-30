import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import save_plot

class ScatterPlot:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df.dropna(inplace=True)
        self.df_only_nb = self.df.iloc[:, 6:]

    def render(self):
        nb_col = nb_row = len(self.df_only_nb.columns)
        plt.style.use('ggplot')
        _, ax = plt.subplots(nb_row, nb_col, figsize=(40, 12), tight_layout=True)
        for i, feature_1 in enumerate(self.df_only_nb):
            for j, feature_2 in enumerate(self.df_only_nb):
                plot = ax[i, j]
                sns.scatterplot(data = self.df, x = feature_2, y = feature_1, hue='Hogwarts House', ax=plot, legend=False)
                if j != 0: plot.yaxis.label.set_visible(False)
                if i != 12: plot.xaxis.label.set_visible(False)
                plot.set_ylabel(plot.get_ylabel(), rotation=20, fontsize=10, labelpad=40)
        save_plot('scatter_plot')
        plt.show()


def main():
    scatter_plot = ScatterPlot('datasets/dataset_train.csv')
    scatter_plot.render()

if __name__ == '__main__':
    main()
