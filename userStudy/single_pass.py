from typing import List
from structure.point import Point
from structure.point_set import PointSet
from structure.hyperplane_set import HyperplaneSet
from structure.hyperplane import Hyperplane
import time
import math


def singlePass(pset: PointSet, u: Point, epsilon, dataset_name):
    start_time = time.time()
    dim = pset.points[0].dim
    utility_range = HyperplaneSet(dim)

    num_of_question = 0
    theta = 5.0/8.0
    S = []
    P = PointSet()
    filter = PointSet()
    data = PointSet(P=pset.points)
    X = []

    p_size = math.ceil(64 * math.log(2 * len(pset.points)))
    for x in data.points:
        if (x.id % 1000) == 0:
            print("# of points processed:  ", x.id)
        # x.print()
        # check if it is pruned
        is_pruned = False
        for FF in S:
            if FF.prune_cone(x, epsilon):
                is_pruned = True
                break
        if is_pruned:
            continue

        # fill in P
        if len(P.points) < p_size:
            P.points.append(x)
            continue

        # add the point to the filter
        left = 0
        right = len(filter.points) - 1
        while left <= right:
            num_of_question += 1
            # print(num_of_question)
            mid = math.floor((left + right) / 2)
            v1 = filter.points[mid].dot_prod(u)
            v2 = x.dot_prod(u)
            if v1 > v2:
                h = Hyperplane(p1=x, p2=filter.points[mid])
                utility_range.hyperplanes.append(h)
                left = mid + 1
                pset.printMiddleSelection(num_of_question, u, "SinglePass",
                                                        dataset_name, filter.points[mid], x, 1, epsilon)
            else:
                h = Hyperplane(p1=filter.points[mid], p2=x)
                utility_range.hyperplanes.append(h)
                right = mid - 1
                pset.printMiddleSelection(num_of_question, u, "SinglePass",
                                                        dataset_name, filter.points[mid], x, 2, epsilon)
            # utility_range.cal_regret(pset, "singlePass", dataset_name, num_of_question)
            # if num_of_question >= 10:
            #    return
            # utility_range.cal_regret_hit_and_run_sample(pset, "singlePass", dataset_name, num_of_question)
            #utility_range.print_time("singlePass", dataset_name, num_of_question, start_time)
        filter.points.insert(left, x)

        # check if there are any points in P that can be pruned by the filter
        PP = PointSet()
        for y in P.points:
            if filter.prune_cone(y, epsilon):
                PP.points.append(y)

        # update
        if len(PP.points) >= theta * len(P.points):
            S.append(filter)
            filter = PointSet()
            P.subtract(PP)

        '''
        print("filter size: ", len(filter.points))
        print("P size: ", len(P.points))
        print("PP size: ", len(PP.points))
        print("S size: ", len(S))
        '''

    if len(filter.points) > 0:
        S.append(filter)
    X.clear()
    for F in S:
        X.append(F.points[0])
    for p in P.points:
        X.append(p)

    # find the best from X
    best = X[0]
    for i in range(1, len(X)):
        num_of_question += 1
        v1 = best.dot_prod(u)
        v2 = X[i].dot_prod(u)
        if v1 < v2:
            h = Hyperplane(p1=best, p2=X[i])
            utility_range.hyperplanes.append(h)
            best = X[i]
            pset.printMiddleSelection(num_of_question, u, "SinglePass", dataset_name, best, X[i], 1, epsilon)
        else:
            h = Hyperplane(p1=X[i], p2=best)
            utility_range.hyperplanes.append(h)
            pset.printMiddleSelection(num_of_question, u, "SinglePass", dataset_name, best, X[i], 2, epsilon)
        # utility_range.cal_regret(pset, "singlePass", dataset_name, num_of_question)
        # if num_of_question >= 10:
        #    return
        # utility_range.cal_regret_hit_and_run_sample(pset, "singlePass", dataset_name, num_of_question)
        #utility_range.print_time("singlePass", dataset_name, num_of_question, start_time)
        

    best.printAlgResult("singlePass", num_of_question, start_time, 0)
    groudtruth = pset.find_top_k(u, 1)[0]
    rr = 1 - best.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    best.printToFile2("singlePass", dataset_name, epsilon, num_of_question, start_time, rr, 10000, 5)

    pset.printFinal(best, num_of_question, u, "SinglePass", dataset_name, epsilon)


def singlePass_generate_traj(pset: PointSet, u: Point, epsilon, dataset_name, index):
    start_time = time.time()
    dim = pset.points[0].dim
    utility_range = HyperplaneSet(dim)

    num_of_question = 0
    theta = 5.0 / 8.0
    S = []
    P = PointSet()
    filter = PointSet()
    data = PointSet(P=pset.points)
    X = []

    p_size = math.ceil(64 * math.log(2 * len(pset.points)))
    for x in data.points:
        if (x.id % 1000) == 0:
            print("# of points processed:  ", x.id)
        # x.print()
        # check if it is pruned
        is_pruned = False
        for FF in S:
            if FF.prune_cone(x, epsilon):
                is_pruned = True
                break
        if is_pruned:
            continue

        # fill in P
        if len(P.points) < p_size:
            P.points.append(x)
            continue

        # add the point to the filter
        left = 0
        right = len(filter.points) - 1
        while left <= right:
            num_of_question += 1
            # print(num_of_question)
            mid = math.floor((left + right) / 2)
            v1 = filter.points[mid].dot_prod(u)
            v2 = x.dot_prod(u)
            if v1 > v2:
                h = Hyperplane(p1=x, p2=filter.points[mid])
                utility_range.hyperplanes.append(h)
                left = mid + 1
                utility_range.print_point_traj(dataset_name, index, num_of_question, filter.points[mid])
            else:
                h = Hyperplane(p1=filter.points[mid], p2=x)
                utility_range.hyperplanes.append(h)
                right = mid - 1
                utility_range.print_point_traj(dataset_name, index, num_of_question, x)
            # utility_range.cal_regret_hit_and_run_sample(pset, "singlePass", dataset_name, num_of_question)
            # utility_range.print_time("singlePass", dataset_name, num_of_question, start_time)
        filter.points.insert(left, x)

        # check if there are any points in P that can be pruned by the filter
        PP = PointSet()
        for y in P.points:
            if filter.prune_cone(y, epsilon):
                PP.points.append(y)

        # update
        if len(PP.points) >= theta * len(P.points):
            S.append(filter)
            filter = PointSet()
            P.subtract(PP)

        '''
        print("filter size: ", len(filter.points))
        print("P size: ", len(P.points))
        print("PP size: ", len(PP.points))
        print("S size: ", len(S))
        '''

    if len(filter.points) > 0:
        S.append(filter)
    X.clear()
    for F in S:
        X.append(F.points[0])
    for p in P.points:
        X.append(p)

    # find the best from X
    best = X[0]
    for i in range(1, len(X)):
        num_of_question += 1
        v1 = best.dot_prod(u)
        v2 = X[i].dot_prod(u)
        if v1 < v2:
            h = Hyperplane(p1=best, p2=X[i])
            utility_range.hyperplanes.append(h)
            best = X[i]
        else:
            h = Hyperplane(p1=X[i], p2=best)
            utility_range.hyperplanes.append(h)
        utility_range.print_point_traj(dataset_name, index, num_of_question, best)
        # utility_range.cal_regret_hit_and_run_sample(pset, "singlePass", dataset_name, num_of_question)
        # utility_range.print_time("singlePass", dataset_name, num_of_question, start_time)


    best.printAlgResult("singlePass", num_of_question, start_time, 0)
    groudtruth = pset.find_top_k(u, 1)[0]
    rr = 1 - best.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    best.printToFile("singlePass", dataset_name, epsilon, num_of_question, start_time, rr)