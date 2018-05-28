import numpy as np
import pylab as plt
import sys


def read_line(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    __l = []
    for line in lines:
        __l.append(line.split(",")[:-1])
    return __l


def mean_number_of_goles(lines):
    __lines = lines[::]
    __v = []
    for line in __lines:
        __v += line
    __v = list(map(int, __v))
    __v = np.array(__v)
    mean = np.mean(__v)
    var = np.var(__v)
    std = np.std(__v)
    print(mean, var, std)


def number_of_goles_histogram(lines):
    __lines = lines[::]

    __v = []
    for line in __lines:
        __v.append(len(line))

    plt.hist(__v)
    plt.show()


def mean_reward(lines):

    __lines = lines[::]

    v = np.zeros(len(__lines[0]))
    for line in __lines:
        line = map(float, line)
        line = list(line)
        v += np.array(line)

    t = range(len(v))
    plt.plot(t, v)
    plt.show()


def mean_reward2(lines):
    __lines = lines[::]

    av_list = []
    std_list = []
    for j in range(len(__lines[0])):
        tmp = []
        for i in range(len(__lines)):
            tmp.append(float(__lines[i][j]))
        tmp = np.array(tmp)

        av_list.append(np.mean(tmp))
        std_list.append(np.std(tmp))

    t = range(len(av_list))
    plt.plot(t, av_list)
    # plt.ylim(0, 1.3)
    # plt.errorbar(t, av_list, yerr=std_list, fmt='', ecolor=None)
    plt.show()


def number_of_goles(lines):
    li = np.array(lines).astype(np.float)
    li = li.transpose()
    m = np.mean(li, axis=1)
    s = np.std(li, axis=1)
    t = range(len(m))
    plt.plot(t, m)
    plt.errorbar(t, m, yerr=s, fmt='', ecolor=None)
    plt.show()


def number_of_goles2(lines1, lines2):
    li = np.array(lines1).astype(np.float)
    li = li.transpose()
    m = np.mean(li, axis=1)
    s = np.std(li, axis=1)
    t = range(len(m))
    # plt.plot(t, m, label="line1")
    plt.errorbar(t, m, yerr=s, fmt='', ecolor=None, label="line1")

    li = np.array(lines2).astype(np.float)
    li = li.transpose()
    m = np.mean(li, axis=1)
    s = np.std(li, axis=1)
    t = range(len(m))
    # plt.plot(t, m, label="line2")
    plt.errorbar(t, m, yerr=s, fmt='', ecolor=None, label="line2")
    plt.legend()
    plt.show()


def main():
    fname1 = "./player.txt"
    fname2 = "./pre_learning.txt"
    # fname = "./player_reward_histry.txt"
    # fname = "./pre_learning_reward_histry.txt"

    # fname = sys.argv[-1]
    # l = read_line(fname)
    # number_of_goles(l)
    # mean_number_of_goles(l)
    # number_of_goles_histogram(l)
    # mean_reward2(l)

    l1 = read_line(fname1)
    l2 = read_line(fname2)
    number_of_goles2(l1, l2)


if __name__ == "__main__":
    main()
