from typing import List
from structure.point import Point
from structure.constant import EQN2
from structure import constant
from structure import frame_fast
from structure.hyperplane import Hyperplane
from structure.hyperplane_set import HyperplaneSet
from structure import others
from structure.point_set import PointSet
import time
import random


"""
 * @brief Get the index of the "current best" point.
 *        Find the average utility vector u of the extreme points of range R. Find the point with the maximum utility w.r.t u
 * @param P         the input car set
 * @param C_idx     the indexes of the current candidate favorite car in P
 * @param ext_vec   the set of extreme vector
 * @return the index of the point in C_idx
"""
def get_current_best_pt(P: List[Point], C_idx: List[int], hset: HyperplaneSet) -> int:
    hset.set_ext_pts()

    # look for the maximum utility point w.r.t. the center utility vector
    best_pt_idx = None
    max_value = float('-inf')
    for idx in C_idx:
        pt = P[idx]
        utility = pt.dot_prod(hset.center)
        if utility > max_value:
            max_value = utility
            best_pt_idx = idx

    return best_pt_idx


"""
 * @brief Generate s cars for selection in a round
 * @param P                     the input car set
 * @param C_idx                 the indexes of the current candidate favorite car in P
 * @param s                     the number of cars for user selection
 * @param current_best_idx      the current best car
 * @param last_best             the best car in previous interaction
 * @param frame                 the frame for obtaining the set of neigbouring vertices of the current best vertex
 *                              (used only if cmp_option = SIMPLEX)
 * @param cmp_option            the car selection mode, which must be either SIMPLEX or RANDOM
 * @return The set of indexes of the points chosen for asking questions
 */
 """
def generate_S(P: List[Point], C_idx: List[int], s, current_best_idx, last_best, frame: List[Point], cmp_option):
    S = []
    if cmp_option == constant.RANDOM:  # RANDOM selection mode
        # randomly select at most s non-overlapping points in the candidate set
        while len(S) < s and len(S) < len(C_idx):
            idx = random.randrange(0, len(C_idx))
            is_new = True
            for i in S:
                if i == idx:
                    is_new = False
                    break
            if is_new:
                S.append(idx)
    elif cmp_option == constant.SIMPlEX:  # SIMPLEX selection mode
        if last_best != current_best_idx or len(frame) == 0:
            # the new frame is not computed before (avoid duplicate frame computation)
            # create one ray for each car in P for computing the frame
            rays = []
            best_i = -1
            for i in range(len(P)):
                if i == current_best_idx:
                    best_i = i
                    continue
                newRay = P[i].__sub__(P[current_best_idx])
                newRay.id = P[i].id
                rays.append(newRay)
            # frame computation
            frame = frame_fast.frameConeFastLP(rays)

        # it is possible that current_best_idx is no longer in the candidate set, then no need to compare again
        for j in range(len(C_idx)):
            if C_idx[j] == current_best_idx:
                S.append(j)
                break

        # select at most s non-overlapping cars in the candidate set based on "neighboring vertices" obtained via frame computation
        for i in range(len(frame)):
            for j in range(len(C_idx)):
                if C_idx[j] == current_best_idx:
                    continue
                if P[C_idx[j]].id == frame[i].id:
                    S.append(j)
                    if len(S) >= s:
                        break
            if len(S) >= s:
                break

        # if less than s car are selected, fill in the remaining one
        if len(S) < s < len(C_idx):
            for j in range(len(C_idx)):
                noIn = True
                for i in range(len(S)):
                    if j == S[i]:
                        noIn = False
                if noIn:
                    S.append(j)
                if len(S) >= s:
                    break
    return S


"""
 * @brief Generate the options for user selection. Update the extreme vectors based on the user feedback
 *        Prune points which are impossible to be top-1
 * @param P                     The skyline dataset
 * @param C_idx                 the indexes of the current candidate favorite car in P
 * @param u                     the utility vector
 * @param s                     the number of cars for user selection
 * @param ext_vec               the set of extreme vecotr
 * @param current_best_idx      the current best car
 * @param last_best             the best car in previous interaction
 * @param frame                 the frame for obtaining the set of neigbouring vertices of the current best vertex
 *                              (used only if cmp_option = SIMPLEX)
 * @param cmp_option            the car selection mode, which must be either SIMPLEX or RANDOM
 */
 """
def update_ext_vec(P: List[Point], C_idx: List[int], u: Point, s, hset: HyperplaneSet, current_best_idx, last_best,
                   frame: List[Point], cmp_option, num_question, dataset_name, epsilon):
    S = generate_S(P, C_idx, s, current_best_idx, last_best, frame, cmp_option)

    max_idx = -1
    max_value = -1
    for i in S:
        pt = P[C_idx[i]]
        value = u.dot_prod(pt)
        if value > max_value:
            max_value = value
            max_idx = i

    value1 = u.dot_prod(P[C_idx[S[0]]])
    value2 = u.dot_prod(P[C_idx[S[1]]])
    pset_org = PointSet()
    if cmp_option == constant.SIMPlEX:
        if value1 > value2:
            pset_org.printMiddleSelection(num_question, u, "UH-Simplex", dataset_name, P[C_idx[S[0]]], P[C_idx[S[1]]], 1, epsilon)
        else:
            pset_org.printMiddleSelection(num_question, u, "UH-Simplex", dataset_name, P[C_idx[S[0]]], P[C_idx[S[1]]], 2, epsilon)

    else:
        if value1 > value2:
            pset_org.printMiddleSelection(num_question, u, "UH-Random", dataset_name, P[C_idx[S[0]]], P[C_idx[S[1]]], 1, epsilon)
        else:
            pset_org.printMiddleSelection(num_question, u, "UH-Random", dataset_name, P[C_idx[S[0]]], P[C_idx[S[1]]], 2, epsilon)

    last_best = current_best_idx
    current_best_idx = C_idx[max_idx]

    for i in S:
        if max_idx == i:
            continue
        tmp = Hyperplane(p1=P[C_idx[i]], p2=P[C_idx[max_idx]])
        C_idx[i] = -1
        hset.hyperplanes.append(tmp)
    hset.set_ext_pts()

    # directly remove the non-favorite point from the candidates
    C_idx = list(filter(lambda x: x != -1, C_idx))

    return C_idx, last_best, current_best_idx


def max_utility(pset: PointSet, u: Point, s, epsilon, maxRound, cmp_option, stop_option, dataset):
    start_time = time.time()
    dim = pset.points[0].dim
    num_question = 0
    rr = 1
    C_idx = [i for i in range(len(pset.points))]
    hset = HyperplaneSet(dim)
    current_best_idx = get_current_best_pt(pset.points, C_idx, hset)

    last_best = -1
    frame = []

    # interaction
    while len(C_idx) > 1 and (rr > epsilon and not others.isZero(rr - epsilon)) and num_question < maxRound:
        num_question += 1
        C_idx.sort()
        C_idx, last_best, current_best_idx = update_ext_vec(pset.points, C_idx, u, s, hset, current_best_idx,
                                                            last_best, frame, cmp_option, num_question, dataset, epsilon)
        if len(C_idx) == 1:
            break
        C_idx, rr = hset.rtree_prune(pset.points, C_idx, stop_option)
        print(f"Round {num_question}: {len(C_idx)} points left, Regret: {rr}")
        '''
        if cmp_option == constant.RANDOM:
            hset.cal_regret(pset, "UH-Random", dataset, num_question)
            # hset.print_time("UH-Random", dataset, num_question, start_time)
        else:
            hset.cal_regret(pset, "UH-Simplex", dataset, num_question)
            if num_question >= 10:
                return
            # hset.print_time("UH-Simplex", dataset, num_question, start_time)
        '''
    # print results
    result = pset.points[get_current_best_pt(pset.points, C_idx, hset)]
    groudtruth = pset.find_top_k(u, 1)[0]
    rr = 1 - result.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    if cmp_option == constant.RANDOM:
        result.printAlgResult("UH-Random", num_question, start_time, 0)
        result.printToFile2("UH-Random", dataset, epsilon, num_question, start_time, rr, 10000, 5)
        pset.printFinal(result, num_question, u, "UH-Random", dataset, epsilon)
    else:
        result.printAlgResult("UH-Simplex", num_question, start_time, 0)
        result.printToFile2("UH-Simplex", dataset, epsilon, num_question, start_time, rr, 10000, 5)
        pset.printFinal(result, num_question, u, "UH-Simplex", dataset, epsilon)

