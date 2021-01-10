def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


if __name__ == '__main__':
    for i in range(2**25):
        if is_power_of_2(i):
            print(i)
