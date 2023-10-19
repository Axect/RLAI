from rlalgs.interface import expected_value, Die, Gaussian

def main():
    # Test Die roll
    die = Die(6)
    print(die)
    print(die.sample_n(100))
    print(expected_value(die, is_even))

    N = Gaussian(0, 1)
    print(N)
    print(N.sample_n(10))

def is_even(x: int) -> float:
    return 1.0 if x % 2 == 0 else 0.0

if __name__ == '__main__':
    main()

