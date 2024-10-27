import itertools

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import torch.optim as optim
from torch.autograd import Variable

import hashlib

import random
from random import randint
import math
import argparse
import pickle
from collections import namedtuple
from itertools import count
import os
import time
from datetime import datetime
import numpy as np

from structure.point import Point
from structure.point_set import PointSet
from structure.constant import EQN2
from structure import constant
from structure.hyperplane import Hyperplane
from structure.hyperplane_set import HyperplaneSet
from structure import others
import time
import random
import heapq


# one_layer_nn and Agent are for insertion
class one_layer_nn(nn.Module):

    def __init__(self, ALPHA, input_size, hidden_size1, output_size, is_gpu=True):
        super(one_layer_nn, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.layer2 = nn.Linear(hidden_size1, output_size, bias=False)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA) # learning rate
        self.loss = nn.MSELoss()
        if is_gpu:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        # self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # sm = nn.Softmax(dim=0)
        sm = nn.SELU() # SELU activation function

        # Variable is used for older torch
        x = Variable(state, requires_grad=False)
        y = self.layer1(x)
        y_hat = sm(y)
        z = self.layer2(y_hat)
        scores = z
        return scores


class AgentLow():
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, input_size, is_gpu, epsEnd=0.1, replace=20):
        # epsEnd was set to be 0.05 initially
        self.GAMMA = gamma  # discount factor
        self.EPSILON = epsilon  # epsilon greedy to choose action
        self.EPS_END = epsEnd  # minimum epsilon
        self.memSize = maxMemorySize  # memory size
        self.batch_size = batch_size
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.action_space_size = action_space_size
        self.Q_eval = one_layer_nn(alpha, input_size, 64, action_space_size, is_gpu)
        self.Q_target = one_layer_nn(alpha, input_size, 64, action_space_size, is_gpu)
        self.state_action_pairs = []

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr = self.memCntr + 1

    def sample(self):
        if len(self.memory) < self.batch_size:
            return self.memory
        else:
            return random.sample(self.memory, self.batch_size)

    def chooseAction(self, observation):
        observation = torch.FloatTensor(observation).to(self.Q_eval.device)
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            # action = torch.argmax(actions).item()
            useless_value, action = torch.max(actions[:self.action_space_size], 0)
        else:
            # action = np.random.choice(self.actionSpace)
            action = randint(0, self.action_space_size - 1)
        self.steps = self.steps + 1
        return int(action)

    def learn(self):
        if len(self.memory) <= 0:
            return

        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        state = []
        action = []
        reward = []
        next_state = []
        for transition in self.sample():
            state.append(list(transition[0]))
            action.append(transition[1])
            reward.append(transition[2])
            next_state.append(list(transition[3]))
        state_action_values = self.Q_eval.forward(torch.FloatTensor(state).to(self.Q_eval.device))
        next_state_action_values = self.Q_target.forward(torch.FloatTensor(next_state).to(self.Q_eval.device))

        # maxA = torch.argmax(Qnext, dim=0)
        max_value, maxA = torch.max(next_state_action_values, 1)  # 1 represents max based on each column
        actions = torch.tensor(action, dtype=torch.int64)
        actions = Variable(actions, requires_grad=False)
        rewards = torch.FloatTensor(reward)
        rewards = Variable(rewards, requires_grad=False)
        state_action_values_target = state_action_values.clone()

        for i in range(len(maxA)):
            temp = rewards[i] + self.GAMMA * (max_value[i])
            state_action_values_target[i, actions[i]] = temp

        if self.steps > 400000:
            if self.EPSILON > self.EPS_END:
                self.EPSILON = self.EPSILON * 0.99
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(state_action_values, state_action_values_target.detach()).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter = self.learn_step_counter + 1

        # be careful of the input format
        # loss = self.Q_eval.loss(Qpred, Qtarget.detach()).to(self.Q_eval.device)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


def lowRL(pset: PointSet, u: Point, epsilon, dataset_name, train=False, trainning_epoch=10000, action_space_size=5):
    start_time = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    dim = pset.points[0].dim
    mp = dim  # number of representative polyhedra
    me = dim  # number of representative extreme points

    input_size = mp * (dim + 1 + dim * me) + action_space_size * dim
    training_epsilon = 0.05
    # init agent
    brain = AgentLow(gamma=0.80, epsilon=0.9, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=action_space_size, input_size=input_size, is_gpu=True)
    if train:  # training
        for epo in range(trainning_epoch):
            t1 = time.time()
            utility_range = HyperplaneSet(dim)
            trainning_sample_u = utility_range.sample_vector()
            trainning_question = 0
            while True:  # stopping condition
                # part of state
                p_candidate, state = utility_range.get_state_low(pset, dim, training_epsilon, me, mp)
                if len(p_candidate) <= 1:
                    break
                # state size mp * (dim + 1 + dim * me)

                # actions
                all_p = p_candidate
                h_cand = []
                if len(all_p) * (len(all_p) - 1) < 2 * action_space_size:
                    for p1, p2 in itertools.combinations(all_p, 2):
                        h_cand.append(Hyperplane(p1=p1, p2=p2))
                    while len(h_cand) < action_space_size:
                        p1, p2 = random.sample(all_p, 2)
                        h_cand.append(Hyperplane(p1=p1, p2=p2))
                else:
                    selected_points = set()
                    while len(h_cand) < action_space_size:
                        p1, p2 = random.sample(all_p, 2)
                        pair = (p1, p2)
                        if pair not in selected_points:
                            h_cand.append(Hyperplane(p1=p1, p2=p2))
                            selected_points.add(pair)
                # action_space_size * dim
                for h in h_cand:
                    state = np.concatenate((state, h.norm))
                action = brain.chooseAction(state)
                brain.state_action_pairs.append([state, action])

                # interaction
                trainning_question += 1
                value1 = trainning_sample_u.dot_prod(h_cand[action].p1)
                value2 = trainning_sample_u.dot_prod(h_cand[action].p2)
                if value1 > value2:
                    h = Hyperplane(p1=h_cand[action].p2, p2=h_cand[action].p1)
                    utility_range.hyperplanes.append(h)
                else:
                    h = Hyperplane(p1=h_cand[action].p1, p2=h_cand[action].p2)
                    utility_range.hyperplanes.append(h)
                utility_range.set_ext_pts()

            # include the last state
            state = np.append(state, np.zeros(action_space_size * dim))
            brain.state_action_pairs.append([state, -1])

            # compute reward
            ind = 0
            while ind < len(brain.state_action_pairs) - 2:
                brain.storeTransition(brain.state_action_pairs[ind][0], brain.state_action_pairs[ind][1],0, brain.state_action_pairs[ind + 1][0])
                ind = ind + 1
            brain.storeTransition(brain.state_action_pairs[ind][0], brain.state_action_pairs[ind][1],100, brain.state_action_pairs[ind + 1][0])
            brain.learn()
            brain.state_action_pairs = []
            print("Episode: ", epo, "Questions: ", trainning_question, "Time: ", time.time() - t1)
            if (epo + 1) % 10000 == 0:
                print("Model saved: " + '_low_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(epo + 1) + '_' + str(action_space_size) + '_' + '.mdl')
                torch.save(brain.Q_eval.state_dict(), '_low_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(epo + 1) + '_' + str(action_space_size) + '_' + '.mdl')
    print("Model training time: ", time.time() - start_time)

    ##################################################
    ##################################################
    # use the trained model to interact
    folder_path = f"../Question_Record/EA_{dataset_name}_{epsilon}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"{folder_path} has been created.")
    else:
        print(f"{folder_path} has already existed. ")

    folder_path += f"/u={u.coord}.txt"

    start_time = time.time()
    brain = AgentLow(gamma=0.80, epsilon=1.0, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=action_space_size, input_size=input_size, is_gpu=False)
    brain.Q_eval.load_state_dict(torch.load('_low_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(trainning_epoch) + '_' + str(action_space_size) + '_' + '.mdl'))
    # device_cpu = torch.device('cpu')  # CPU 设备
    # brain.Q_eval.to(device_cpu)
    # 保存模型的权重到一个新的文件，不包括其他状态
    # torch.save(brain.Q_eval.state_dict(), 'cpu_low_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(trainning_epoch) + '_' + str(action_space_size) + '_' + '.mdl')
    brain.EPSILON = 0
    num_question = 0
    result = None

    utility_range = HyperplaneSet(dim)
    while True:  # stopping condition
        # part of state
        p_candidate, state = utility_range.get_state_low(pset, dim, epsilon, me, mp)
        if len(p_candidate) <= 1:
            result = p_candidate[0]
            break
        # state size mp *  me * dim

        # actions
        all_p = p_candidate
        h_cand = []
        if len(all_p) * (len(all_p) - 1) < 2 * action_space_size:
            for p1, p2 in itertools.combinations(all_p, 2):
                h_cand.append(Hyperplane(p1=p1, p2=p2))
            while len(h_cand) < action_space_size:
                p1, p2 = random.sample(all_p, 2)
                h_cand.append(Hyperplane(p1=p1, p2=p2))
        else:
            selected_points = set()
            while len(h_cand) < action_space_size:
                p1, p2 = random.sample(all_p, 2)
                pair = (p1, p2)
                if pair not in selected_points:
                    h_cand.append(Hyperplane(p1=p1, p2=p2))
                    selected_points.add(pair)
        # action_space_size * dim
        for h in h_cand:
            state = np.concatenate((state, h.norm))
        action = brain.chooseAction(state)
        brain.state_action_pairs.append([state, action])

        # interaction
        num_question += 1
        print("Interaction: ", num_question, len(p_candidate))
        value1 = u.dot_prod(h_cand[action].p1)
        value2 = u.dot_prod(h_cand[action].p2)
        if value1 > value2:
            h = Hyperplane(p1=h_cand[action].p2, p2=h_cand[action].p1)
            utility_range.hyperplanes.append(h)
            pset.printMiddleSelection(num_question, u, "EA", dataset_name, h_cand[action].p1, h_cand[action].p2, 1, epsilon)
        else:
            h = Hyperplane(p1=h_cand[action].p1, p2=h_cand[action].p2)
            utility_range.hyperplanes.append(h)
            pset.printMiddleSelection(num_question, u, "EA", dataset_name, h_cand[action].p1, h_cand[action].p2, 2, epsilon)
        utility_range.set_ext_pts()
        # utility_range.cal_regret(pset, "EA", dataset_name, num_question)
        # utility_range.print_time("EA", dataset_name, num_question, start_time)

    # print results
    result.printAlgResult("EA", num_question, start_time, 0)
    best = pset.find_top_k(u, 1)[0]
    rr = 1 - result.dot_prod(u) / best.dot_prod(u)
    print("Regret: ", rr)
    result.printToFile2("EA", dataset_name, epsilon, num_question, start_time, rr, trainning_epoch, action_space_size)

    pset.printFinal(result, num_question, u, "EA", dataset_name, epsilon)