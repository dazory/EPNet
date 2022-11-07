import os
import random
import math


def main():
    out_dir = '/ws/data/carla/ImageSets'
    split = ['train', 'val']
    TRAIN, VAL = 0, 1
    ratio = [8, 2]

    start_idx = 0
    end_idx = 1025

    idx_list = []
    for idx in range(start_idx, end_idx+1, 1):
        idx_list.append("%06d" % idx)

    print("idx_list:")
    print(idx_list[:10])
    print(idx_list[-10:])
    print('')

    random.shuffle(idx_list)
    print("shuffled idx_list:")
    print(idx_list[:10])
    print(idx_list[-10:])
    print('')

    num_idx = len(idx_list)
    num_train = math.ceil(num_idx * ratio[TRAIN] / (ratio[TRAIN] + ratio[VAL]))
    num_val = num_idx - num_train
    print('num_idx: ', num_idx)
    print('num_train: ', num_train)
    print('num_val: ', num_val)
    print('')

    with open(f"{out_dir}/{split[TRAIN]}.txt", 'w') as file:
        cnt_train = 0
        for idx in idx_list[:num_train]:
            file.write(f"{idx}\n")
            cnt_train += 1
        print("cnt_train: ", cnt_train)

    with open(f"{out_dir}/{split[VAL]}.txt", 'w') as file:
        cnt_val = 0
        for idx in idx_list[:num_val]:
            file.write(f"{idx}\n")
            cnt_val += 1
        print("cnt_val: ", cnt_val)

    print(f"cnt_total: ", cnt_train + cnt_val)


if __name__ == '__main__':
    main()