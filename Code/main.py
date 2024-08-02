from structure.hyperplane import Hyperplane
from structure.point import Point
from structure.point_set import PointSet
import uh
import single_pass
import random
from structure import constant
import time
import highRL
import lowRL
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments with different parameters.")
    parser.add_argument('params', nargs='+', help="List of parameters")
    args = parser.parse_args()

    alg_name = str(args.params[0])
    dataset_name = str(args.params[1])
    epsilon = float(args.params[2])
    trainning_size = int(args.params[3])
    action_size = int(args.params[4])

    '''
    with open("../config.txt", 'r') as config:
        alg_name, dataset_name, epsilon = config.readline().split()
        epsilon = float(epsilon)
    '''

    # dataset
    pset = PointSet(f'{dataset_name}.txt')
    dim = pset.points[0].dim
    # pset = pset.skyline()
    for i in range(len(pset.points)):
        pset.points[i].id = i

    '''
        # write skyline data to txt file for backup
        with open(f'input/{dataset_name}Skyline.txt', 'w') as file:
            file.write(f'{len(pset.points)} {pset.points[0].dim}\n')
            for p in pset.points:
                file.writelines(f'{value:.6f} ' for value in p.coord)
                file.write('\n')
        '''

    '''
    # utility vector
    u = Point(dim)
    utility = list(map(float, config.readline().split()))
    for i in range(dim):
        u.coord[i] = float(utility[i])
    '''

    '''
        sum_coord = 0.0
        for i in range(dim):
            u.coord[i] = random.random()
            sum_coord += u.coord[i]
        for i in range(dim):
            u.coord[i] = u.coord[i] / sum_coord
        # for i in range(dim):
        #    u.coord[i] = 1.0 / dim
        '''

    u = Point(dim)
    for i in range(dim):
        u.coord[i] = float(args.params[5 + i])
    print(f"{alg_name}, {dataset_name}, {epsilon}, {trainning_size}, {action_size}, {u.coord}")

    # ground truth
    top_set = pset.find_top_k(u, 1)
    top_set[0].printAlgResult("GroundTruth", 0, time.time(), 0)

    if alg_name == "UH-Random":  # the UH-Random algorithm
        uh.max_utility(pset, u, 2, epsilon, 1000, constant.RANDOM, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "UH-Simplex":  # the UH-Simplex algorithm
        uh.max_utility(pset, u, 2, epsilon, 1000, constant.SIMPlEX, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "singlePass":  # the singlePass algorithm
        single_pass.singlePass(pset, u, epsilon, dataset_name)
    elif alg_name == "singlePass_traj":  # the singlePass algorithm
        single_pass.singlePass_generate_traj(pset, u, epsilon, dataset_name, 1)
    elif alg_name == "highRL":
        highRL.highRL(pset, u, epsilon, dataset_name, True, trainning_size, action_size)
    elif alg_name == "lowRL":
        lowRL.lowRL(pset, u, epsilon, dataset_name, True, trainning_size, action_size)

