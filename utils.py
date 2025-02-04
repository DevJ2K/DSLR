from Colors import RED, BHRED, RESET
import os, sys
import matplotlib.pyplot as plt
import numpy as np

def print_error(e: Exception) -> None:
    line_size = 60
    print(BHRED, "=" * line_size, RESET, sep="", file=sys.stderr)
    print(f"{BHRED}Raise Exception Name: {RED}{type(e).__name__}{RESET}", file=sys.stderr)
    if (str(e) != ""):
        print(f"{BHRED}Description: {RED}{str(e)}{RESET}", file=sys.stderr)

    if (e.__context__ is not None):
        print(f"{BHRED}Initial Cause Name: {RED}{type(e.__context__).__name__}{RESET}", file=sys.stderr)
        print(f"{BHRED}Initial Cause Description: {RED}{e.__context__}{RESET}", file=sys.stderr)
    print(BHRED, "=" * line_size, RESET, sep="", file=sys.stderr)



def save_plot(filename: str) -> None:
    try:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig(f'plots/{filename}.png')
    except Exception as e:
        print_error(e)
        exit(1)

def min_max_scaling(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val)

def display_loss_plot(losses):
    fig, ax = plt.subplots(2, 2, tight_layout=True)
    fig.suptitle('Loss Function')
    for i in range(4):
        subplot = ax[int(i / 2), i % 2]
        subplot.set_title(f'Class {i}')
        subplot.plot(losses[i], c='r')


def display_accuracy_score_plot(scores):
    fig, ax = plt.subplots(2, 2, tight_layout=True)
    fig.suptitle('Accuracy Score')
    for i in range(4):
        subplot = ax[int(i / 2), i % 2]
        subplot.set_title(f'Class {i}')
        subplot.plot(scores[i])