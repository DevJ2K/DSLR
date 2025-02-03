import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from Colors import *
from utils import min_max_scaling

pd.set_option('future.no_silent_downcasting', True)

class LogisticRegression:
    def __init__(self, x_train, y, learning_rate=0.1, epochs=3000):
        self.X = x_train
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(self.X.shape[1]) * 0.01
        self.bias = 0
        self.losses = []
        self.thetas = []

    def sigmoid_function(self, x: float) -> float:
        return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))

    def binary_cross_entropy(self, y_true, y_pred): # Loss function
        return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

    def gradient_descent(self, y_true, y_pred):
        diff = y_pred - y_true
        gradient_bias = np.mean(diff)
        gradients_weights = np.dot(self.X.T, diff) / len(self.X)
        return gradients_weights, gradient_bias

    def update_model_parameters(self, error_weights, error_bias):
        self.weights -= self.learning_rate * error_weights
        self.bias -= self.learning_rate * error_bias

    def fit(self):
        for i in range(4):
            y = np.where(self.y == i, 1, 0)
            self.weights = np.random.randn(self.X.shape[1]) * 0.01
            self.bias = 0
            losses = []
            for epoch in range(self.epochs):
                x_dot_weights = np.dot(self.weights, self.X.T) + self.bias
                # print(x_dot_weights)
                pred = np.array([self.sigmoid_function(x) for x in x_dot_weights])
                # print(pred)
                error_weights, error_bias = self.gradient_descent(y, pred)
                # print(self.weights)
                # print(error_weights, error_bias)
                losses.append(self.binary_cross_entropy(y, pred))
                self.update_model_parameters(error_weights, error_bias)
                # print(error_weights)
                # print(self.weights)
                # print(pred)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {losses[-1]}")
            self.losses.append((losses, i))
            return
            self.thetas.append((self.weights, i))
            # predi = self.predict(self.X)
            # print('predi: ', predi, len(predi))

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        proba = np.array([self.sigmoid_function(x) for x in x_dot_weights])
        return proba


def main():
    # parser = argparse.ArgumentParser(description="Use a train dataset to generate a file containing the weights that will be used for the prediction.")
    df = pd.read_csv('datasets/dataset_train.csv')
    df.dropna(inplace=True)
    classes = { value: index for index, value in enumerate(df['Hogwarts House'].unique()) }
    print(classes)
    y = df.replace({'Hogwarts House': classes})['Hogwarts House']
    x_train = df.select_dtypes('float')
    x_train = min_max_scaling(x_train)
    # print(x_train)
    log_reg = LogisticRegression(x_train=x_train, y=y)
    log_reg.fit()
    # print(log_reg.predict(log_reg.X))
    # print(log_reg.losses)
    plt.plot(log_reg.losses[0][0])
    plt.show()

if __name__ == "__main__":
    main()
