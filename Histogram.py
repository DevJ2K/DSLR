import matplotlib.pyplot as plt
import pandas as pd
from utils import save_plot

class Histogram:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df.dropna(inplace=True)
        self.df_only_nb = self.df.iloc[:, 6:]

    def render(self):
        nb_col = 4
        nb_row = 4
        plt.style.use('ggplot')
        _, ax = plt.subplots(nb_row, nb_col, tight_layout=True, figsize=(15, 8))
        for i, col in enumerate(self.df_only_nb):
            plot = ax[int(i / nb_row), i % nb_col]
            for key, grp in self.df.groupby(['Hogwarts House']):
                plot.hist(grp[col], alpha=0.5, label = key)
            plot.set_title(col, fontsize=10)
        ax[nb_row-1, nb_col-1].axis('off')
        ax[nb_row-1, nb_col-2].axis('off')
        ax[nb_row-1, nb_col-3].axis('off')
        handles, labels = ax[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc='best', borderpad=1.5, prop={'size': 12})
        ax[nb_row-1, nb_col-3].set_title('Hogwarts students repartition by course', fontsize=20, x=1, y=0.3, fontweight='medium')
        save_plot('histogram')
        plt.show()


def main():
    histogram = Histogram('datasets/dataset_train.csv')
    histogram.render()


if __name__ == '__main__':
    main()
