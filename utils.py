from Colors import RED, BHRED, RESET
import os, sys
import matplotlib.pyplot as plt

def print_error(e: Exception) -> None:
    print(f"{BHRED}Name: {RED}{type(e).__name__}{RESET}")
    print(f"{BHRED}Description: {RED}{str(e)}{RESET}")


def save_plot(filename: str) -> None:
    try:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig(f'plots/{filename}.png')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        exit(1)
