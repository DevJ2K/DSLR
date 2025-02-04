import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse, pickle
from Colors import *
from utils import min_max_scaling, display_accuracy_score_plot, display_loss_plot

pd.set_option('future.no_silent_downcasting', True)

class LogisticRegression:
    def __init__(self, X, y, learning_rate=0.1, epochs=500):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(self.X.shape[1])
        self.bias = 0
        self.losses = []
        self.accuracy_scores = []
        self.models = []

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
        for i in range(len(self.y.unique())):
            y = np.where(self.y == i, 1, 0)
            self.weights = np.zeros(self.X.shape[1])
            self.bias = 0
            losses = []
            accuracy_score = []
            for epoch in range(self.epochs):
                x_dot_weights = np.dot(self.weights, self.X.T) + self.bias
                pred = np.array([self.sigmoid_function(x) for x in x_dot_weights])
                error_weights, error_bias = self.gradient_descent(y, pred)
                losses.append(self.binary_cross_entropy(y, pred))
                self.update_model_parameters(error_weights, error_bias)
                accuracy_score.append(self.accuracy_score(y, self.binary_prediction()))
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {losses[-1]}")
            self.losses.append(losses)
            self.accuracy_scores.append(accuracy_score)
            self.models.append((self.weights.copy(), self.bias))

    def binary_prediction(self):
        x_dot_weights = np.dot(self.X, self.weights) + self.bias
        proba = np.array([self.sigmoid_function(value) for value in x_dot_weights])
        return [1 if p >= 0.5 else 0 for p in proba]

    def predict(self):
        probabilities = []
        for weights, bias in self.models:
            x_dot_weights = np.dot(self.X, weights) + bias
            proba = np.array([self.sigmoid_function(value) for value in x_dot_weights])
            probabilities.append(proba)
        probabilities = np.array(probabilities)
        return np.argmax(probabilities, axis=0)
    
    def save_weights(self, filename="weights.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Weights saved in {filename}")

    def load_weights(self, filename="weights.pkl"):
        with open(filename, "rb") as f:
            self.models = pickle.load(f)

    def global_accuracy_score(self, y_true, y_pred):
        return sum(y_pred == y_true) / len(y_true)


def main():
    # parser = argparse.ArgumentParser(description="Use a train dataset to generate a file containing the weights that will be used for the prediction.")
    df = pd.read_csv('datasets/dataset_train.csv')
    df.dropna(inplace=True)
    classes = { value: index for index, value in enumerate(df['Hogwarts House'].unique()) }
    y = df.replace({'Hogwarts House': classes})['Hogwarts House']
    x_train = df.select_dtypes('float')
    x_train = min_max_scaling(x_train)
    # print(x_train)
    log_reg = LogisticRegression(X=x_train, y=y)
    log_reg.fit()
    log_reg.save_weights()
    # log_reg.load_weights()
    y_pred = log_reg.predict()
    # log_reg.accuracy_score(y_true=y, y_pred=y_pred)
    display_accuracy_score_plot(log_reg.accuracy_scores)
    display_loss_plot(log_reg.losses)
    plt.show()

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
