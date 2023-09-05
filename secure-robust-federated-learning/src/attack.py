# BSD 2-Clause License

# Copyright (c) 2022, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
    This file contains the attack functions on Federated Learning.
'''

import copy
import numpy as np
import random
from robust_estimator import krum
from scipy.spatial import distance
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from scipy.linalg import qr
import matplotlib.pyplot as plt
import csv



# Model Poisoning Attack--Training benign model at malicious agent.
def benign_train(mal_train_loaders, network, criterion, optimizer, params_copy, device):
    local_grads = []
    for p in list(network.parameters()):
        local_grads.append(np.zeros(p.data.shape))

    for idx, (feature, _, target, true_label) in enumerate(mal_train_loaders, 0):
        feature = feature.to(device)
        target = target.type(torch.long).to(device)
        true_label = true_label.type(torch.long).to(device)
        optimizer.zero_grad()
        output = network(feature)
        loss_val = criterion(output, true_label)
        loss_val.backward()
        optimizer.step()
    
    for idx, p in enumerate(network.parameters()):
        local_grads[idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

    with torch.no_grad():
        for idx, p in enumerate(list(network.parameters())):
            p.copy_(params_copy[idx])

    return local_grads

# Model Poisoning Attack--Return the gradients at previous round.
def est_accuracy(mal_visible, t, path):
    delta_other_prev = None
    if len(mal_visible) >= 1:
        mal_prev_t = mal_visible[-1]
        delta_other_prev = np.load('./checkpoints/' + path + 'ben_delta_t%s.npy' % mal_prev_t, allow_pickle=True)
        delta_other_prev /= (t - mal_prev_t)
    return delta_other_prev

# Model Poisoning Attack--Compute the weight constrain loss.
def weight_constrain(loss1, network, constrain_weights, t, device):
    params = list(network.parameters())
    loss_fn = nn.MSELoss(size_average=False, reduce=True)
    start_flag = 0

    for idx in range(len(params)):
        grad = torch.from_numpy(constrain_weights[idx]).to(device)
        if start_flag == 0:
            loss2 = loss_fn(grad, params[idx])
        else:
            loss2 += loss_fn(grad, params[idx])
        start_flag = 1
    rho = 1e-3
    loss = loss1 + loss2 * rho

    return loss

# Model Poisoning Attack--The main function for MPA.
def mal_single(mal_train_loaders, train_loaders, network, criterion, optimizer, params_copy, device, mal_visible, t, dist=True, mal_boost=1, path=None):
    start_weights = params_copy.copy()
    constrain_weights = []

    for p in list(network.parameters()):
        constrain_weights.append(np.zeros(p.data.shape))

    delta_other_prev = est_accuracy(mal_visible, t, path)

    # Add benign estimation
    if len(mal_visible) >= 1:
        for idx in range(len(start_weights)):
            delta_other = torch.from_numpy(delta_other_prev[idx]).to(device)
            start_weights[idx].data.sub_(delta_other)
    
    # Load shared weights for malicious agent
    with torch.no_grad():
        for idx, p in enumerate(list(network.parameters())):
            p.copy_(start_weights[idx])

    final_delta = benign_train(mal_train_loaders, network, criterion, optimizer, start_weights, device)
    for idx, p in enumerate(start_weights):
        constrain_weights[idx] = p.data.cpu().numpy() - final_delta[idx] / 10

    delta_mal = []
    delta_local = []
    for p in list(network.parameters()):
        delta_mal.append(np.zeros(p.data.shape))
        delta_local.append(np.zeros(p.data.shape))
    
    for idx, (feature, target) in enumerate(train_loaders, 0):
        feature = feature.to(device)
        target = target.type(torch.long).to(device)
        optimizer.zero_grad()
        output = network(feature)
        loss_val = criterion(output, target)
        loss = weight_constrain(loss_val, network, constrain_weights, t, device)
        loss.backward()
        optimizer.step()

    for idx, p in enumerate(list(network.parameters())):
        delta_local[idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

    for i in range(int(len(train_loaders) / len(mal_train_loaders) + 1)):
        for idx, (feature, mal_data, _, target) in enumerate(mal_train_loaders, 0):
            mal_data = mal_data.to(device)
            target = target.type(torch.long).to(device)
            output = network(mal_data)
            loss_mal = criterion(output, target)

            optimizer.zero_grad()
            loss_mal.backward()
            optimizer.step()

    # Boost the malicious data gradients.
    for idx, p in enumerate(list(network.parameters())):
        delta_mal[idx] = (params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy() - delta_local[idx]) * mal_boost + delta_local[idx]

    return delta_mal


# Trimmed Mean Attack--Main function for TMA.
def attack_trimmedmean(network, local_grads, mal_index, b=2):
    benign_max = []
    benign_min = []
    average_sign = []
    mal_param = []
    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    for p in list(network.parameters()):
        benign_max.append(np.zeros(p.data.shape))
        benign_min.append(np.zeros(p.data.shape))
        average_sign.append(np.zeros(p.data.shape))
        mal_param.append(np.zeros(p.data.shape))
    for idx, p in enumerate(average_sign):
        for c in range(len(local_param)):
            average_sign[idx] += local_param[c][idx]
        average_sign[idx] = np.sign(average_sign[idx])
    for idx, p in enumerate(network.parameters()):
        temp = []
        for c in range(len(local_param)):
            local_param[c][idx] = p.data.cpu().numpy() - local_param[c][idx]
            temp.append(local_param[c][idx])
        temp = np.array(temp)
        benign_max[idx] = np.amax(temp, axis=0)
        benign_min[idx] = np.amin(temp, axis=0)
    
    for idx, p in enumerate(average_sign):
        for aver_sign, b_max, b_min, mal_p in np.nditer([p, benign_max[idx], benign_min[idx], mal_param[idx]], op_flags=['readwrite']):
            if aver_sign < 0:
                if b_min > 0:
                    mal_p[...] = random.uniform(b_min/b, b_min)
                else:
                    mal_p[...] = random.uniform(b_min*b, b_min)
            else:
                if b_max > 0:
                    mal_p[...] = random.uniform(b_max, b_max*b)
                else:
                    mal_p[...] = random.uniform(b_max, b_max/b)
    for c in mal_index:
        for idx, p in enumerate(network.parameters()):
            local_grads[c][idx] = -mal_param[idx] + p.data.cpu().numpy()
    return local_grads


# Krum Attack--Main function for KA.
def attack_krum(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3):

    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size

    average_sign = np.zeros(list(network.parameters())[param_index].data.shape)
    benign_max = np.zeros(list(network.parameters())[param_index].data.shape)

    for c in range(len(local_param)):
        average_sign += local_param[c][param_index]
    average_sign  = np.sign(average_sign)
    min_dis = np.inf
    max_dis = -np.inf
    for i in range(m):
        if i in mal_index:
            continue
        else:
            temp_min_dis = 0
            temp_max_dis = 0
            for j in range(m):
                if j in mal_index or j == i:
                    continue
                else:
                    temp_min_dis += distance.euclidean(local_grads[i][param_index].flatten(), local_grads[j][param_index].flatten())
        temp_max_dis += distance.euclidean(local_grads[i][param_index].flatten(), benign_max.flatten())

        if temp_min_dis < min_dis:
            min_dis = temp_min_dis
        if temp_max_dis > max_dis:
            max_dis = temp_max_dis
    
    upper_bound = 1.0
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * average_sign
        _, choose_index = krum(krum_local, f=1)
        if choose_index in mal_index:
            print('found a lambda')
            break
        elif lambda1 < lower_bound:
            print(choose_index, 'Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0
    
    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * average_sign

    return local_grads



def attack_krum_improved(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3):

    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size

    average_sign = np.zeros(list(network.parameters())[param_index].data.shape)
    benign_max = np.zeros(list(network.parameters())[param_index].data.shape)

    for c in range(len(local_param)):
        average_sign += local_param[c][param_index]
    average_sign = np.sign(average_sign)
    min_dis = np.inf
    max_dis = -np.inf
    for i in range(m):
        if i in mal_index:
            continue
        else:
            temp_min_dis = 0
            temp_max_dis = 0
            for j in range(m):
                if j in mal_index or j == i:
                    continue
                else:
                    temp_min_dis += distance.euclidean(local_grads[i][param_index].flatten(), local_grads[j][param_index].flatten())
            temp_max_dis += distance.euclidean(local_grads[i][param_index].flatten(), benign_max.flatten())

            if temp_min_dis < min_dis:
                min_dis = temp_min_dis
            if temp_max_dis > max_dis:
                max_dis = temp_max_dis
    
    # Try different attack strengths within the specified range
    upper_bound = 1.0
    lower_bound = 1e-12  # Experiment with lower bound
    step_size = (upper_bound - lower_bound) / 10  # Divide range into steps
    for lambda1 in np.arange(upper_bound, lower_bound, -step_size):
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * average_sign
        _, choose_index = krum(krum_local, f=1)
        
        if choose_index in mal_index:
            print('found a lambda:', lambda1)
            break
    
    # Apply the selected attack strength to malicious gradients
    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * average_sign

    return local_grads




def bulyan_attack_krum(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3, target_layer=0, target_idx=0):

    benign_max = []
    attack_vec = []

    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size
    for p in list(network.parameters()):
        benign_max.append(np.zeros(p.data.shape))
        attack_vec.append(np.zeros(p.data.shape))

    for idx, p in enumerate(attack_vec):
        for c in range(len(local_param)):
            if c == target_layer and idx == target_idx:
                attack_vec[idx] += 1
 
    upper_bound = 1.0
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * attack_vec[param_index]
        _, choose_index = krum(krum_local, f=1)
        if choose_index in mal_index:
            break
        elif lambda1 < lower_bound:
            print(choose_index, 'Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0
    
    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * attack_vec[param_index]
    
    return local_grads

def backdoor(network, train_loader, test_loader, threshold=90, device='cpu', lr=1e-4, batch_size=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    acc = 0.0
    attack_acc = 0.0
    while (acc < threshold) or (attack_acc < threshold):
        for _, (feature, target) in enumerate(train_loader, 0):
            if np.random.randint(2) == 0:
                clean_feature = (feature.to(device)).view(-1, 784)
                clean_target = target.type(torch.long).to(device)
                optimizer.zero_grad()
                output = network(clean_feature)
                loss = criterion(output, clean_target)
                loss.backward()
                optimizer.step()
            else:
                attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                attack_target = torch.zeros(batch_size, dtype=torch.long).to(device)
                optimizer.zero_grad()
                output = network(attack_feature)
                loss = criterion(output, attack_target)
                loss.backward()
                optimizer.step()

        correct = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature = (feature.to(device)).view(-1, 784)
                target = target.type(torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        acc = 100. * correct / len(test_loader.dataset)
        print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))

        correct = 0
        # attack success rate
        with torch.no_grad():
            for feature, target in test_loader:
                feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                target = torch.zeros(batch_size, dtype=torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        attack_acc = 100. * correct / len(test_loader.dataset)
        print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))
        print(acc, attack_acc)

def attack_xie(local_grads, weight, choices, mal_index):
    attack_vec = []
    for i, pp in enumerate(local_grads[0]):
        tmp = np.zeros_like(pp)
        for ji, j in enumerate(choices):
            if j not in mal_index:
                tmp += local_grads[j][i]
        attack_vec.append((-weight) * tmp / len(choices))
    for i in mal_index:
        local_grads[i] = attack_vec
    return local_grads




def add_gaussian_noise(w, scale):
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    else:
        for k in w_attacked.keys():
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    return w_attacked


def change_weight(w_attack, w_honest, change_rate=0.5):
    w_result = copy.deepcopy(w_honest)
    device = w_attack[list(w_attack.keys())[0]].device
    for k in w_honest.keys():
        w_h = w_honest[k]
        w_a = w_attack[k]

        assert w_h.shape == w_a.shape

        honest_idx = torch.FloatTensor((np.random.random(w_h.shape) > change_rate).astype(np.float)).to(device)
        attack_idx = torch.ones_like(w_h).to(device) - honest_idx

        weight = honest_idx * w_h + attack_idx * w_a
        w_result[k] = weight

    return w_result





'''
# -*-*--*--*--*--*--*--*--*--*-

***  Define the CSI attack and selection ***

# -*-*--*--*--*--*--*--*--*--*-

'''

def Update_CSI_CPI(H, selected_clients, victim_idx, attacker_id, conspirator_id):
    """
    Replace victim client with a new client with a larger effective channel gain
    and smaller interference component, while maintaining spatial compatibility.

    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - selected_clients: list of indices of clients currently active
    - victim_idx: index of victim client to be replaced
    - attacker_id: index of attacker client to replace victim
    - conspirator_id: index of conspirator client whose CSI should be orthogonal to attacker's CSI
    
    Returns:
    - H_new: updated channel matrix with victim replaced by attacker
    """   
    # Decompose victim's channel into effective channel and interference
    gv = H[victim_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(selected_clients)):
        if selected_clients[j] != victim_idx:
            gj = H[selected_clients[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    ev = gv - ev
    
    # Generate replacement channel for victim_idx-1
    alpha_v = np.random.normal(loc=1.5, scale=0.1)
    omega_v = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients)-1,))
    # new_victim_channel = alpha_v * H[victim_idx-1, :] + omega_v.dot(ev)
    
    # Generate replacement channel for attacker_id
    alpha_a = np.random.normal(loc=1.5, scale=0.1)
    omega_a = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients),))
    ev_consp = np.zeros_like(H[conspirator_id, :])
    for j in range(len(selected_clients)):
        if selected_clients[j] != conspirator_id:
            gj = H[selected_clients[j], :]
            ev_consp += np.vdot(H[conspirator_id, :], gj) / np.vdot(gj, gj) * gj
    ev_a = gv - ev_consp
    new_attacker_channel = alpha_a * H[victim_idx, :] + omega_a.dot(ev_a)
    # new_attacker_channel = alpha_a * H[victim_idx, :] + omega_a.reshape((9,1)).dot(ev_a)

    # Replace victim with attacker
    H_new = np.copy(H)
    H_new[attacker_id, :] = new_attacker_channel
    
    return H_new





def Update_CSI(H, selected_clients, victim_idx, attacker_id):
    """
    Replace victim client with a new client with a larger effective channel gain
    and smaller interference component, while maintaining spatial compatibility.
    
    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - selected_clients: list of indices of clients currently active
    - victim_idx: index of victim client to be replaced
    - attacker_id: index of attacker client to replace victim
    
    Returns:
    - H_new: updated channel matrix with victim replaced by attacker
    """
    
    # Decompose victim's channel into effective channel and interference
    gv = H[victim_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(selected_clients)):
        if selected_clients[j] != victim_idx:
            gj = H[selected_clients[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    ev = gv - ev
    
    # Generate replacement channel with larger effective channel gain and smaller interference
    alpha = np.random.normal(loc=1.5, scale=0.1)
    omega = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients),))
    new_client_channel = alpha * gv + omega.dot(ev)
    
    # Replace victim with attacker
    H_new = np.copy(H)
    H_new[attacker_id, :] = new_client_channel
    
    return H_new


def find_victim(conspirator_idx, S0, H):
    result =[]
    for idx in reversed(S0):
        victim_idx = idx
        gv = H[victim_idx, :]
        gc = H[conspirator_idx, :]
        
        # Calculate the interference and orthogonal components for the victim and conspirator
        interference_victim = np.abs(np.vdot(gv, gc))
        orthogonal_victim = np.linalg.norm(gv - interference_victim)
        
        interference_conspirator = np.abs(np.vdot(gc, gv))
        orthogonal_conspirator = np.linalg.norm(gc - interference_conspirator)
        
        # Compare the conspirator and victim based on orthogonal and interference components
        if orthogonal_conspirator > orthogonal_victim and interference_conspirator < interference_victim:
            result.append(victim_idx) 

    # If no suitable victim is found, return None
    if len(result) == 0:
        return S0[9]
    else:
        return result[-1]


def calculate_effective_channel_gains(H):
    gains = []
    for idx in range(H.shape[0]):
        g = H[idx, :]
        g_norm = np.linalg.norm(g)
        gains.append(g_norm)
    return gains

# gains = calculate_effective_channel_gains(H)
# conspirator_idx = np.argmax(gains)

def client_rank(client_gain, gains):
    sorted_gains = sorted(gains, reverse=True)
    return sorted_gains.index(client_gain)


def categorize_clients(gains, categories):
    categorized_clients = {i: [] for i in range(categories)}
    gain_min, gain_max = min(gains), max(gains)
    step = (gain_max - gain_min) / categories

    for idx, gain in enumerate(gains):
        category = int((gain - gain_min) // step)
        if category == categories:
            category -= 1
        categorized_clients[category].append(idx)

    return categorized_clients

# categories = 10
# categorized_clients = categorize_clients(gains, categories)

# rank_improvements = []

# for category in range(categories):
#     conspirator_candidates = categorized_clients[category]
#     improvements = []

#     for conspirator_idx in conspirator_candidates:
#         original_rank = client_rank(gains[conspirator_idx], gains)

        # Modify the selection algorithm or other factors to improve the rank of the conspirator
        # For example, you can increase the conspirator's channel gain by a certain factor





def Update_CSI_CPE(H,K,N,conspirator_idx,malicious_idx,alpha,omega):
    """
    The malicious attacker is involved to help the conspirator. 
    # first we get the clients list 
    # and then we observe the clients list,we get some prospective conspirators.we let attacker replace j-1 th user,and based on this new 
    # clients subset,we can get the client selected  based on j-1 users(including attacker)


    ## t-1 round's the client list: [0,1,2..j-1,j,..N] (0<j<=N) victim j is based on the 0~(j-1) CSI
    ## pre-t round: we set CSI of attakcer to replace j-1,then the promsing list of j-1 users: [0,1,2..j-2,attacker]
    ## t round: when we choose the j-th user,the user is based on the new j-1 users list.We can the feature of the new j-th user.
    ## or we can get a list of promsing consipirators.
    
    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - K: K  proposed users 
    - j: index of victim client in the predict list(based on plain-text)
    - attacker_id: index of attacker client to replace victim
    - conspirator_id: index of conspirator with high probablity of being selected in next round
    
    Returns:
    - H_new: updated channel matrix of selecting conspirator in next round with high probablity
    """

    H_new = np.copy(H)
    orthogonal_values = {}
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        orthogonal_values[s_hat_n] = g_norms[idx]
         # Sort S_0 based on orthogonal_values
    S0_ranked = sorted(S_0, key=lambda x: orthogonal_values[x], reverse=True)
    # print("S0_ranked:",S0_ranked[:20])
    
    # Find the indices of the (j-1)-th and j-th users in S0_ranked
    
    victim_idx = find_victim(conspirator_idx, S0_ranked, H)
    # print("victim_idx is:",victim_idx)
    j_minus_1_idx = S0_ranked[S0_ranked.index(victim_idx)-1] 
    # print("victim_idx-1 is:",j_minus_1_idx)




    # Generate replacement channel with larger effective channel gain and smaller interference => move to the args list
    # alpha = np.random.normal(loc=1.0, scale=0.1)
    # omega = np.random.normal(loc=0.3, scale=0.1, size=(len(selected_clients),))

    # Calculate the orthogonal part of the conspirator's channel
    gc = H[conspirator_idx, :]
    interference_conspirator = np.zeros_like(gc)
    
    for j in S0_ranked:
        if j != conspirator_idx:
            gj = H[j, :]
            interference_conspirator += np.vdot(gc, gj) / np.vdot(gj, gj) * gj
    orthogonal_conspirator = gc - interference_conspirator
    
    # Calculate the interference part of the victim's channel
    gv = H[victim_idx, :]
    interference_victim = np.zeros_like(gv)

    for j in S0_ranked:
        if j != victim_idx:
            gj = H[j, :]
            interference_victim += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    orthogonal_victim = gv - interference_victim
    influence_channel_victim = alpha * orthogonal_victim + omega*interference_victim



     # Decompose victim-1's channel into effective channel and interference
    gv = H[j_minus_1_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(S0_ranked)):
        if S0_ranked[j] != j_minus_1_idx:
            gj = H[S0_ranked[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # ev = gv - ev
    new_client_channel = alpha * gv + omega*ev
    

    # Calculate the distance between the conspirator and the victim
    distance = abs(S0_ranked.index(conspirator_idx) - S0_ranked.index(victim_idx))

    # distance =  abs(conspirator_idx - S0_ranked.index(victim_idx))

    # Create an empty array for the influence channels
    influence_channels = np.zeros_like(H[0, :])

    # Calculate the influence channels for clients between the conspirator and the victim
    for i in range(1, distance):
        idx = S0_ranked[S0_ranked.index(victim_idx) + i]
        gv = H[idx, :]
        interference = np.zeros_like(gv)

        for j in S0_ranked:
            if j != idx:
                gj = H[j, :]
                interference += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
        orthogonal = gv - interference
        influence_channel = alpha * orthogonal + omega*interference
        influence_channels += influence_channel
    
    # Create the new feedback ha for the malicious user  orthogonal_conspirator + interference_victim +new_client_channel + new_client_channel_victim
   
    # we can influence a  a specific client and change its rank(from a higher --> lower rank)  + influence_channels
    ha = influence_channel_victim + new_client_channel  + influence_channels
    # we improve the selection probablity  of a specific client's client
    # ha = new_client_channel + orthogonal_conspirator+ interference_conspirator
    
    # Replace the (j-1)-th user with the malicious user in the channel matrix
    H_new[malicious_idx, :] = ha
    
    # return H_new,victim_idx,j_minus_1_idx
    return H_new








def Update_CSI_CPE_2(H,K,N,conspirator_idx,malicious_idx):
    """
    The malicious attacker is involved to help the conspirator. 
    # first we get the clients list 
    # and then we observe the clients list,we get some prospective conspirators.we let attacker replace j-1 th user,and based on this new 
    # clients subset,we can get the client selected  based on j-1 users(including attacker)


    ## t-1 round's the client list: [0,1,2..j-1,j,..N] (0<j<=N) victim j is based on the 0~(j-1) CSI
    ## pre-t round: we set CSI of attakcer to replace j-1,then the promsing list of j-1 users: [0,1,2..j-2,attacker]
    ## t round: when we choose the j-th user,the user is based on the new j-1 users list.We can the feature of the new j-th user.
    ## or we can get a list of promsing consipirators.
    
    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - K: K  proposed users 
    - j: index of victim client in the predict list(based on plain-text)
    - attacker_id: index of attacker client to replace victim
    - conspirator_id: index of conspirator with high probablity of being selected in next round
    
    Returns:
    - H_new: updated channel matrix of selecting conspirator in next round with high probablity
    """

    H_new = np.copy(H)
    orthogonal_values = {}
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        orthogonal_values[s_hat_n] = g_norms[idx]
         # Sort S_0 based on orthogonal_values
    S0_ranked = sorted(S_0, key=lambda x: orthogonal_values[x], reverse=True)
    # print("S0_ranked:",S0_ranked[:20])
    
    # Find the indices of the (j-1)-th and j-th users in S0_ranked
    
    victim_idx = find_victim(conspirator_idx, S0_ranked, H)
    # print("victim_idx is:",victim_idx)
    j_minus_1_idx = S0_ranked[S0_ranked.index(victim_idx)-1] 
    # print("victim_idx-1 is:",j_minus_1_idx)




    # Generate replacement channel with larger effective channel gain and smaller interference
    alpha = np.random.normal(loc=1.2, scale=0.1)
    omega = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients),))


    # h_attacker = np.zeros_like(H[malicious_idx, :])
    # for client in S0_ranked:
    #     S0_ranked.remove(client)
    #     gv = H[client, :]
    #     ev = np.zeros_like(gv)
    #     for j in range(len(S0_ranked)):
    #         if S0_ranked[j] != client:
    #             gj = H[S0_ranked[j], :]
    #             ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    #     # ev = gv - ev
    #             new_client_channel = alpha * gv + omega.dot(ev)
    #             h_attacker  =  h_attacker + new_client_channel





    # S0_ranked.remove(victim_idx)
    # S0_ranked.remove(j_minus_1_idx)
    # S0_ranked.append(malicious_idx)

    # Calculate the orthogonal part of the conspirator's channel
    gc = H[conspirator_idx, :]
    interference_conspirator = np.zeros_like(gc)
    
    for j in S0_ranked:
        if j != conspirator_idx:
            gj = H[j, :]
            interference_conspirator += np.vdot(gc, gj) / np.vdot(gj, gj) * gj
    orthogonal_conspirator = gc - interference_conspirator
    
    # Calculate the interference part of the victim's channel
    gv = H[victim_idx, :]
    interference_victim = np.zeros_like(gv)

    for j in S0_ranked:
        if j != victim_idx:
            gj = H[j, :]
            interference_victim += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    orthogonal_victim = gv - interference_victim
    influence_channel_victim = alpha * orthogonal_victim + omega.dot(interference_victim)



     # Decompose victim-1's channel into effective channel and interference
    gv = H[j_minus_1_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(S0_ranked)):
        if S0_ranked[j] != j_minus_1_idx:
            gj = H[S0_ranked[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # ev = gv - ev
    new_client_channel = alpha * gv + omega.dot(ev)
    
    # Replace victim with attacker
    # H_new = np.copy(H)
    # H_new[attacker_id, :] = new_client_channel

    # gv = H[victim_idx, :]
    # ev = np.zeros_like(gv)
    # for j in range(len(S0_ranked)):
    #     if S0_ranked[j] != victim_idx:
    #         gj = H[S0_ranked[j], :]
    #         ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # ev = gv - ev
    # new_client_channel_victim = alpha * gv + omega.dot(ev)

        # Calculate the distance between the conspirator and the victim
    distance = abs(S0_ranked.index(conspirator_idx) - S0_ranked.index(victim_idx))

    # Create an empty array for the influence channels
    influence_channels = np.zeros_like(H[0, :])

    # Calculate the influence channels for clients between the conspirator and the victim
    for i in range(1, distance):
        idx = S0_ranked[S0_ranked.index(victim_idx) + i]
        gv = H[idx, :]
        interference = np.zeros_like(gv)

        for j in S0_ranked:
            if j != idx:
                gj = H[j, :]
                interference += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
        orthogonal = gv - interference
        influence_channel = alpha * orthogonal + omega.dot(interference)
        influence_channels += influence_channel

    # Calculate the new feedback ha for the malicious user
    # ha = influence_channels

    # # Replace the (j-1)-th user with the malicious user in the channel matrix
    # H_new[malicious_idx, :] = ha



    # # Calculate the interference part of the victim's channel
    # gv = H[S0_ranked[10], :]
    # interference_10 = np.zeros_like(gv)

    # for j in S0_ranked:
    #     if j != S0_ranked[10]:
    #         gj = H[j, :]
    #         interference_10 += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # orthogonal_10 = gv - interference_10
    # influence_channel_10 = alpha * orthogonal_10 + omega.dot(interference_10)


    #     # Calculate the interference part of the victim's channel
    # gv =  H[S0_ranked[11], :]
    # interference_11 = np.zeros_like(gv)

    # for j in S0_ranked:
    #     if j != S0_ranked[11]:
    #         gj = H[j, :]
    #         interference_11 += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # orthogonal_11 = gv - interference_11
    # influence_channel_11 = alpha * orthogonal_11 + omega.dot(interference_11)

    



    
    # Create the new feedback ha for the malicious user  orthogonal_conspirator + interference_victim +new_client_channel + new_client_channel_victim
   
    # we can influence a  a specific client and change its rank(from a higher --> lower rank)
    ha = influence_channel_victim + new_client_channel + influence_channels
     # we improve the selection probablity  of a specific client's client
    # ha = new_client_channel + orthogonal_conspirator+ interference_conspirator
    
    # Replace the (j-1)-th user with the malicious user in the channel matrix
    H_new[malicious_idx, :] = ha
    
    return H_new,victim_idx,j_minus_1_idx











def Plain_Select_CPE(H, K, N):
    selected_clients = []
    noise_power = 1  # Gaussian noise power
    
    # Spatial compatibility quantization algorithm
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    orthogonal_values = {}
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        orthogonal_values[s_hat_n] = g_norms[idx]
         # Sort S_0 based on orthogonal_values
        S_0_ranked = sorted(S_0, key=lambda x: orthogonal_values[x], reverse=True)

        
        # Check for aligned users and select one with higher gain and lower interference
        for j in S_0:
            if j != s_hat_n:
                if np.abs(np.dot(H[s_hat_n], H[j].conj().T)) > 0.99:
                    alpha = np.dot(H[s_hat_n], H[j].conj().T)
                    gi_norms_j = np.linalg.norm(H[j])
                    gi_norms_selected = np.linalg.norm(H[selected_clients])
                    omega = np.sqrt(1 - np.abs(alpha)**2) * gi_norms_j / gi_norms_selected
                    new_client_channel = alpha * H[s_hat_n] + omega * H[j]
                    if np.linalg.norm(new_client_channel) > np.linalg.norm(H[s_hat_n]):
                        s_hat_n = j
                else:
                    continue
        selected_clients.append(s_hat_n)

        # Compute precoding vector and SINR for selected user
        C = np.linalg.pinv(H[selected_clients])
        m = np.ones((len(selected_clients), 1)) / np.sqrt(len(selected_clients))  # Unit power constraint
        interference = 0.0
        for j in S_0:
            if j != s_hat_n:
                interference += np.abs(np.dot(H[s_hat_n], H[j].conj().T)) ** 2 / np.dot(H[j], H[j].conj().T)
        noise = np.random.randn() * np.sqrt(noise_power)
        SINR = np.abs(np.dot(H[s_hat_n], C[:, i]) * m[i]) ** 2 / (interference + noise)

    return selected_clients








def Plain_Select(H, K, N):
    selected_clients = []
    noise_power = 1  # Gaussian noise power
    
    # Spatial compatibility quantization algorithm
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        
        # Check for aligned users and select one with higher gain and lower interference
        for j in S_0:
            if j != s_hat_n:
                if np.abs(np.dot(H[s_hat_n], H[j].conj().T)) > 0.99:
                    alpha = np.dot(H[s_hat_n], H[j].conj().T)
                    gi_norms_j = np.linalg.norm(H[j])
                    gi_norms_selected = np.linalg.norm(H[selected_clients])
                    omega = np.sqrt(1 - np.abs(alpha)**2) * gi_norms_j / gi_norms_selected
                    new_client_channel = alpha * H[s_hat_n] + omega * H[j]
                    if np.linalg.norm(new_client_channel) > np.linalg.norm(H[s_hat_n]):
                        s_hat_n = j
                else:
                    continue
        selected_clients.append(s_hat_n)

        # Compute precoding vector and SINR for selected user
        C = np.linalg.pinv(H[selected_clients])
        m = np.ones((len(selected_clients), 1)) / np.sqrt(len(selected_clients))  # Unit power constraint
        interference = 0.0
        for j in S_0:
            if j != s_hat_n:
                interference += np.abs(np.dot(H[s_hat_n], H[j].conj().T)) ** 2 / np.dot(H[j], H[j].conj().T)
        noise = np.random.randn() * np.sqrt(noise_power)
        SINR = np.abs(np.dot(H[s_hat_n], C[:, i]) * m[i]) ** 2 / (interference + noise)

    return selected_clients



   

        


# Replace one client per iteration
# replace attack
# victim_idx = np.random.choice(range(num_selected_clients))
# victim_idx = np.random.randint(num_selected_clients)
# print(victim_idx)
'''
victim_idx = 0
new_selected_clients = replace(selected_clients,H, victim_idx)
print("New---- clients:",new_selected_clients)

# The client selection process use case
# Parameters
N = 10  # Number of base station antennas
K = 100  # Total number of clients
P = 100  # Total transmit power


# H = np.random.randn(K, N) + 1j * np.random.randn(K, N)


# Generate channel matrix with random complex entries
H = np.random.randn(K, N) + 1j * np.random.randn(K, N)
# Compute the total transmit power across all antennas
total_power = np.sum(np.abs(H)**2)
# Normalize the channel matrix to satisfy the power constraint
H = H * np.sqrt(P / total_power)


# print("attack 1----------------------")
selected_clients = Plain_Select(H,K,N)
print(selected_clients)

H_new  =  Update_CSI(H,selected_clients,selected_clients[-1],18)


# The formal client selection process:


new_selected_clients = Plain_Select(H_new,K,N)
print(new_selected_clients)













print("attack 2----------------------")

# The client selection process use case
# Parameters
N = 100  # Number of base station antennas
K = 100  # Total number of clients
P = 100  # Total transmit power


# H = np.random.randn(K, N) + 1j * np.random.randn(K, N)


# Generate channel matrix with random complex entries
H = np.random.randn(K, N) + 1j * np.random.randn(K, N)
# Compute the total transmit power across all antennas
total_power = np.sum(np.abs(H)**2)
# Normalize the channel matrix to satisfy the power constraint
H = H * np.sqrt(P / total_power)


# def Update_CSI_CPE(H,K,N,attacker_id,conspirator_id):
#Predict the clients based on Plain text
selected_clients = Plain_Select(H,K,N)
print("Original selected_clients is:",selected_clients[:10])

#Then replace the (j-1)th user

# H_flattened = H.flatten()
# sorted_indices = np.argsort(np.abs(H_flattened))
# sorted_indices = sorted_indices.tolist()
# print(sorted_indices.index(28))


before = []
after = []
for conspirator in range(10,100):
    before.append(conspirator)
    H_new,victim_idx,j_minus_1_idx  =  Update_CSI_CPE(H,K,N,selected_clients[conspirator],18)
    # print("The conspirator is:",selected_clients[conspirator])
    new_selected_clients = Plain_Select(H_new,K,N)
    # print(new_selected_clients[:10])
    # print("Before repalce,the index of conspirator is",selected_clients.index(selected_clients[conspirator]))
    # print("After repalce,the index of conspirator is: ",new_selected_clients.index(selected_clients[conspirator]))
    after.append(new_selected_clients.index(selected_clients[conspirator]))
    # Write train_acc and test_accs to a CSV file
    with open('replace_result_3.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Count', 'Before CPE', 'After CPE','victim_idx','j_minus_1_idx'])
        for epoch in range(len(before)):
            writer.writerow([epoch+1, before[epoch], after[epoch],victim_idx,j_minus_1_idx])
    

plt.plot(before, label='Before CPE')
plt.plot(after, label='After CPE')
plt.xlabel('Count')
plt.ylabel('Rank')
plt.savefig('rank_conspirator_3.pdf')
plt.legend()
plt.show()

'''

# we can replace a specific client so that it is never selected 

# we can influence a  a specific client and change its rank(from a higher --> lower rank)

# we improve the selection probablity  of a specific client's client


# we observer the influence of power of attacker on the attack success rate






