import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse, pickle
from Colors import *
from utils import *

pd.set_option('future.no_silent_downcasting', True)

class LogisticRegression:
    def __init__(self, X, y, learning_rate, epochs):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(self.X.shape[1])
        self.bias = 0.0
        self.losses = []
        self.accuracy_scores = []
        self.models = []

    def sigmoid_function(self, x: float) -> float:
        return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: # Loss function
        return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

    def gradient_descent(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
        diff = y_pred - y_true
        gradient_bias = np.mean(diff)
        gradients_weights = np.dot(self.X.T, diff) / len(self.X)
        return gradients_weights, gradient_bias

    def update_model_parameters(self, error_weights: list, error_bias: float) -> None:
        self.weights -= self.learning_rate * error_weights
        self.bias -= self.learning_rate * error_bias

    def fit(self) -> None:
        print(f'{BYELLOW}[Logistic Regression: Multi-Classifier]{RESET} \
            \n{GREENB}{BWHITE}[TRAINING PHASE]{RESET}')
        for i in range(len(self.y.unique())):
            y = np.where(self.y == i, 1, 0)
            self.weights = np.zeros(self.X.shape[1])
            self.bias = 0
            losses = []
            accuracy_score = []
            print(f'\n{MAGB}{BWHITE}[MODEL {i+1}]{RESET}')
            for epoch in range(self.epochs):
                x_dot_weights = np.dot(self.weights, self.X.T) + self.bias
                pred = np.array([self.sigmoid_function(x) for x in x_dot_weights])
                error_weights, error_bias = self.gradient_descent(y, pred)
                losses.append(self.binary_cross_entropy(y, pred))
                self.update_model_parameters(error_weights, error_bias)
                accuracy_score.append(self.accuracy_score(y, self.binary_prediction()))
                if epoch % (self.epochs / 10) == 0:
                    print(f'{BCYAN}Epoch {epoch}    {BHRED}Loss: {losses[-1]:.4f}{RESET}')
            self.losses.append(losses)
            self.accuracy_scores.append(accuracy_score)
            self.models.append((self.weights.copy(), self.bias))

    def binary_prediction(self) -> list:
        x_dot_weights = np.dot(self.X, self.weights) + self.bias
        proba = np.array([self.sigmoid_function(value) for value in x_dot_weights])
        return [1 if p >= 0.5 else 0 for p in proba]

    def predict(self) -> np.ndarray:
        probabilities = []
        for weights, bias in self.models:
            x_dot_weights = np.dot(self.X, weights) + bias
            proba = np.array([self.sigmoid_function(value) for value in x_dot_weights])
            probabilities.append(proba)
        probabilities = np.array(probabilities)
        return np.argmax(probabilities, axis=0)

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return sum(y_pred == y_true) / len(y_true)

    def save_weights(self, filename="weights.pkl") -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.models, f)
        print_info(f'Weights saved in {UGREEN}{filename}')

    def load_weights(self, filename="weights.pkl") -> None:
        with open(filename, "rb") as f:
            self.models = pickle.load(f)

def main():
    parser = argparse.ArgumentParser(
        description='Use a train dataset to generate a file containing the weights '
        'that will be used for the prediction.'
    )
    parser.add_argument(
        '-dataset',
        type=str,
        default='datasets/dataset_train.csv',
        help='Path to a train datatest file to train the model'
    )
    parser.add_argument(
        '-epochs',
        type=int,
        default=500,
        help='Total number of iterations of all the training data '
        'in one cycle for training the model'
    )
    parser.add_argument(
        '-learning-rate',
        type=int,
        default=0.1,
        help='Hyperparameter that controls how much to change '
        'the model when the model weights are updated.'
    )
    try:
        args = parser.parse_args()
        try:
            df = pd.read_csv(args.dataset)
        except Exception:
            print(f"{BHRED}Fail to read file '{RED}{args.dataset}{BHRED}'.{RESET}")
            raise
        df.dropna(inplace=True)
        classes = { value: index for index, value in enumerate(df['Hogwarts House'].unique()) }
        y = df.replace({'Hogwarts House': classes})['Hogwarts House']
        x_train = df.select_dtypes('float64')
        x_train = min_max_scaling(x_train)
        log_reg = LogisticRegression(
            X=x_train,
            y=y,
            epochs=args.epochs,
            learning_rate=args.learning_rate)
        log_reg.fit()
        log_reg.save_weights()
        display_accuracy_score_plot(log_reg.accuracy_scores)
        display_loss_plot(log_reg.losses)
        plt.show()
    except Exception as e:
        print_error(e)

if __name__ == "__main__":
    main()

    # df = pd.read_csv('datasets/dataset_test.csv')
    # df = df.select_dtypes('float')
    # df.dropna(axis=1, how='all', inplace=True)
    # # print(df)
    # df = min_max_scaling(df)
    # log_reg = LogisticRegression(X=df, y=[])
    # log_reg.load_weights()
    # # print(log_reg.models)
    # print('predi: ', log_reg.predict())
