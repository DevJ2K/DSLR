from Colors import RED, BHRED, RESET


def print_error(e: Exception) -> None:
    print(f"{BHRED}Name: {RED}{type(e).__name__}{RESET}")
    print(f"{BHRED}Description: {RED}{str(e)}{RESET}")

