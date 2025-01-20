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

# Standard Deviation
print("=== STD(Standard Deviation) ===")
my_std = (10 - my_mean) ** 2 + (13 - my_mean) ** 2 + (9 - my_mean) ** 2
my_std /= 2
my_std = my_std ** 0.5
print(my_std)

# min
# ...


def percentile_calc(tab: list[int], percentage: int) -> float:
    tab.sort()
    percentage /= 100
    pos = percentage * (len(tab) - 1)
    rest_pos = pos - math.floor(pos)
    int_pos = math.floor(pos)
    # print(pos)
    print(int_pos)
    print(rest_pos)

    if int_pos + 1 == len(tab):
        next_pos = int_pos
    else:
        next_pos = int_pos + 1

    print(next_pos)
    nb = tab[next_pos] - tab[int_pos]
    nb_to_add = nb * rest_pos
    print(nb, nb_to_add)
    return tab[int_pos] + nb_to_add


# 25%
print("=== PERCENTILE : 25%")
print(percentile_calc([10, 13, 9], 25))

# 50%
print("=== PERCENTILE : 50%")
print(percentile_calc([10, 13, 9], 50))

# print(my_mean )

# 75%
print("=== PERCENTILE : 75%")
print(percentile_calc([10, 13, 9], 75))

# max
# ...