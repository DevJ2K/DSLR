import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import save_plot

class PairPlot:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df.dropna(inplace=True)
        self.df.drop('Index', axis=1, inplace=True)

    def render(self):
        plt.style.use('ggplot')
        sns.pairplot(self.df, hue='Hogwarts House')
        save_plot('pair_plot')
        plt.show()


def main():
    pair_plot = PairPlot('datasets/dataset_train.csv')
    pair_plot.render()

if __name__ == '__main__':
    main()
