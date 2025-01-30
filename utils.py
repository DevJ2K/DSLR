from Colors import RED, BHRED, RESET
import os, sys
import matplotlib.pyplot as plt

def print_error(e: Exception) -> None:
    line_size = 60
    print(BHRED, "=" * line_size, RESET, sep="")
    print(f"{BHRED}Raise Exception Name: {RED}{type(e).__name__}{RESET}")
    if (str(e) != ""):
        print(f"{BHRED}Description: {RED}{str(e)}{RESET}")

    if (e.__context__ is not None):
        print(f"{BHRED}Initial Cause Name: {RED}{type(e.__context__).__name__}{RESET}")
        print(f"{BHRED}Initial Cause Description: {RED}{e.__context__}{RESET}")
    print(BHRED, "=" * line_size, RESET, sep="")



def save_plot(filename: str) -> None:
    try:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig(f'plots/{filename}.png')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        exit(1)
