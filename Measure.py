class Measure:
    headtop2chin = 25  # cm，大多数成年人的头颅长度
    eye2mouth = 1.618 * headtop2chin / (2 * 2.618)  # cm，按照常规尺寸测算的眼睛到嘴的距离

    def __init__(self):
        pass


if __name__ == '__main__':
    print(Measure.eye2mouth)
    pass
