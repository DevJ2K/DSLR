import pandas as pd
import math

data = [[10, 18, 11], [13, 15, 8, 4], [9, 20, 3]]

df = pd.DataFrame(data)

print(df.describe())

# count
# len data[...][0]


print("=== MEAN ===")
my_mean = (10 + 13 + 9) / 3
print(my_mean)



def variance(values: tuple) -> float:
    """
    Calculates and returns the variance of a tuple
    """
    mean = sum(values) / len(values)
    return sum(((xi - mean) ** 2 for xi in values)) / (len(values) - 1)

def standard_deviation(values: tuple) -> float:
    return variance(values) ** 0.5

# Standard Deviation
print("=== STD(Standard Deviation) ===")
my_std = (10 - my_mean) ** 2 + (13 - my_mean) ** 2 + (9 - my_mean) ** 2
my_std /= 3 - 1
my_std = my_std ** 0.5
print(my_std)

print(standard_deviation([13, 9, 10]))
print(variance([13, 9, 10]))

# min
# ...




def percentile_calc1(tab: list[int], percentage: int) -> float:
    tab.sort()
    percentage /= 100
    pos = percentage * (len(tab) - 1)
    rest_pos = pos - math.floor(pos)
    int_pos = math.floor(pos)
    # print(pos)
    # print(int_pos)
    # print(rest_pos)

    if int_pos + 1 == len(tab):
        next_pos = int_pos
    else:
        next_pos = int_pos + 1

    # print(next_pos)
    nb = tab[next_pos] - tab[int_pos]
    nb_to_add = nb * rest_pos
    # print(nb, nb_to_add)
    return tab[int_pos] + nb_to_add


def percentile_calc2(tab: list[int], percentage: int) -> float:
    tab.sort()
    if percentage == 25:
        n = 1
    elif percentage == 50:
        n = 2
    elif percentage == 75:
        n = 3
    return(tab[len(tab) // 4 * n])
    # print(tab[(len(tab) // 4) * 3])

def median(values: list) -> int:
    """
    Calculates and returns the median of a list
    """
    values.sort()
    length = len(values)
    if length % 2 == 0:
        median1 = values[length // 2]
        median2 = values[length // 2 - 1]
        return (median1 + median2) / 2
    else:
        return values[length // 2]

print([10, 13, 9], [9, 10, 13])
# 25%
print("=== PERCENTILE : 25%")
print(percentile_calc1([10, 13, 9], 25))
print(percentile_calc2([10, 13, 9], 25))

# 50%
print("=== PERCENTILE : 50%")
print(percentile_calc1([10, 13, 9, 14], 50))
print(median([10, 13, 9, 14]))

# print(my_mean )

# 75%
print("=== PERCENTILE : 75%")
print(percentile_calc1([10, 13, 9], 75))
print(percentile_calc2([10, 13, 9], 75))

# max
# ...
