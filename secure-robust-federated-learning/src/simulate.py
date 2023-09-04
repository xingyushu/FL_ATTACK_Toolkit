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
    The simulation program for Federated Learning.
'''

import argparse
from attack import attack_krum, attack_trimmedmean, attack_xie, backdoor, mal_single
from data import MalDataset
from networks import ConvNet
import numpy as np
import random
from robust_estimator import krum, filterL2, median, trimmed_mean, bulyan, ex_noregret, mom_filterL2, mom_ex_noregret, mom_krum
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import utils as vutils
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import time
import copy
import csv
from attack import Plain_Select,Update_CSI,Update_CSI_CPE

random.seed(2022)
np.random.seed(5)
torch.manual_seed(11)

MAL_FEATURE_FILE = './data/%s_mal_feature_10.npy'
MAL_TARGET_FILE = './data/%s_mal_target_10.npy'
MAL_TRUE_LABEL_FILE = './data/%s_mal_true_label_10.npy'

ALL_METHODS = [
    'random', 'PlainCSI', 'PlainCSI_ATTACK','maxrate','FL_TDoS','FL_CPE'
]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--nworker', type=int, default=100)
    parser.add_argument('--perround', type=int, default=10)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--round', type=int, default=50) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=10.)
    parser.add_argument('--buckets', type=int, default=10)

    #client slection args.

    parser.add_argument('--method', type=str, default='random', help='client selection',
                        choices=ALL_METHODS)

    #random selection
    parser.add_argument('--random',action = 'store_true',help = 'use random selection')

    # Plain_CSI selection
    parser.add_argument('--PlainCSI',action = 'store_true',help = 'use Plain_CSI selection')
    parser.add_argument('--p', type=int, default=1000,
                        help="total power limit: p")
    parser.add_argument('--B', type=int, default=1000,
                        help="total bandwidth limit: B")


    # Plain_CSI selection ATTACK args
    parser.add_argument('--PlainCSI_ATTACK',action = 'store_true',help = 'use Plain_CSI ATTACK selection')
    parser.add_argument('--victim',type=int, default=9,help = 'victim index in the Estimated selected clients in TDoS attack,victim <= N-1 ')
    parser.add_argument('--conspirator_idx',type=int, default=9,help = 'conspirator index in the Estimated selected clients in CPE attack,conspirator <= N-1 ')
    parser.add_argument('--attacker_index',type=int, default=18,help = 'attacker index in the H,attacker_index <= K')

    parser.add_argument('--FL_TDoS',action = 'store_true',help = 'use FL_TDoS selection')
    parser.add_argument('--FL_CPE',action = 'store_true',help = 'use FL_CPE selection')



    # Power-d arguments
    parser.add_argument('--power_d',action = 'store_true',
                        help = 'use Pow-d selection')
    parser.add_argument('--d',type = int,default = 30,
                        help='d in Pow-d selection')

    # Active Federated Learning arguments
    parser.add_argument('--afl',action = 'store_true',
                        help = 'use AFL selection')


    # Malicious agent setting
    parser.add_argument('--malnum', type=int, default=30)
    parser.add_argument('--agg', default='average', help='average, ex_noregret, filterl2, krum, median, trimmedmean, bulyankrum, bulyantrimmedmean, bulyanmedian, mom_filterl2, mom_ex_noregret, iclr2022_bucketing, icml2021_history, clustering')
    parser.add_argument('--attack', default='noattack', help="noattack, trimmedmean, krum, backdoor, modelpoisoning, xie")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu") 
    # mal_index = list(range(args.malnum))
    mal_index = [18,55] 

    if args.dataset == 'MNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        # batch_size = len(train_set) // args.nworker
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform))
        mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('mnist'), MAL_TRUE_LABEL_FILE%('mnist'), MAL_TARGET_FILE%('mnist'), transform=transform), batch_size=args.batchsize)

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    elif args.dataset == 'Fashion-MNIST':

        train_set = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = torchvision.transforms.ToTensor())
        # batch_size = len(train_set) // args.nworker
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = torchvision.transforms.ToTensor()))
        mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('fashion'), MAL_TRUE_LABEL_FILE%('fashion'), MAL_TARGET_FILE%('fashion'), transform=torchvision.transforms.ToTensor()), batch_size=args.batchsize)

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    # Split into multiple training set
    local_size = len(train_set) // args.nworker
    sizes = []
    sum = 0
    for i in range(0, args.nworker):
        sizes.append(local_size)
        sum = sum + local_size
    sizes[0] = sizes[0] + len(train_set)  - sum
    train_sets = random_split(train_set, sizes)
    train_loaders = []
    for trainset in train_sets:
        train_loaders.append(DataLoader(trainset, batch_size=args.batchsize, shuffle=True))

    # define training loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=args.lr)
    # prepare data structures to store local gradients
    local_grads = []
    for i in range(args.nworker):
        local_grads.append([])
        for p in list(network.parameters()):
            local_grads[i].append(np.zeros(p.data.shape))
    prev_average_grad = None

    print('Malicious node indices:', mal_index, 'Attack Type:', args.attack)

    file_name = './results/' + args.attack + '_' + args.agg + '_' + args.dataset + '_' + str(len(mal_index)) + '_' + str(time.strftime("%Y%m%d"))+ '.txt'
    txt_file = open(file_name, 'w') 




    for round_idx in range(args.round):
        print("Round: ", round_idx)
        print("Client Selection Procss-------------------")

        if  args.method == 'PlainCSI':
    
                print("This is the  CSI-base Max Rate selection")
                
                N = args.perround   # Number of base station antennas  --num_users
                K = args.nworker  # Total number of clients
                H = np.random.randn(K,N) + 1j * np.random.randn(K, N)
                
                P = args.p  # Total transmit power
                B = args.B  # Total transmit bandwidth

                # Generate channel matrix with random complex entries
                H = np.random.randn(K, N) + 1j * np.random.randn(K, N)

                # Compute the total transmit power across all antennas
                total_power = np.sum(np.abs(H)**2)

                # Normalize the channel matrix to satisfy the power constraint
                H = H * np.sqrt(P / total_power)

                # Generate a list of clients with random values for ID, bandwidth, and power
                client_list = []
                for i in range(K):
                    client = {
                        'id': i,
                        'bandwidth': random.randint(1, 100),  # Assign a random bandwidth value for each client
                        'power': random.randint(1, 100)
                    }
                    client_list.append(client)

                selected_clients = []
                remaining_bandwidth = B

                for client in client_list:
                    # Check if adding the client satisfies the bandwidth constraint
                    if remaining_bandwidth >= client['bandwidth']:
                        choices = Plain_Select(H, K, N)
                        remaining_bandwidth -= client['bandwidth']
                    else:
                        break

                print("The PlainCSI Selected Clients are:",choices)




        elif  args.method == 'FL_TDoS':
                print("this is the FL_TDoS selection ATTACK")
                
                N = args.perround   # Number of base station antennas  --num_users
                K = args.nworker  # Total number of clients
                P = args.p  # Total transmit power
                B = args.B  # Total transmit bandwidth

                # Generate channel matrix with random complex entries
                H = np.random.randn(K, N) + 1j * np.random.randn(K, N)

                # Compute the total transmit power across all antennas
                total_power = np.sum(np.abs(H)**2)

                # Normalize the channel matrix to satisfy the power constraint
                H = H * np.sqrt(P / total_power)
                # print(H)
                selected_clients = Plain_Select(H,K,N)
                print("Original selected_clients is:",selected_clients)

                index = args.victim
                attacker_index =  args.attacker_index
                # The ture clients after attack
                H_new  =  Update_CSI(H,selected_clients,selected_clients[index],attacker_index)
                choices = Plain_Select(H_new,K,N)
           
                print("Attack clients by FL_TDoS are:",choices)

            #    with open('./results/selected_TDoS_clients_' + args.attack + '_' + args.agg + '_' + args.dataset + '_' + str(len(mal_index)) + '_' + str(time.strftime("%Y%m%d"))+ '.csv', 'a+', newline='') as csvfile:
            #         writer = csv.writer(csvfile)
            #         # writer.writerow(['Original Client ID', 'Attack Client ID'])
            #         writer.writerow([iter+1,index,attacker_index,list(selected_clients), list(idxs_users)])

                # with open('selected_clients_data_TDoS.csv', 'a+', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     # writer.writerow(['Original Client ID', 'Attack Client ID'])
                #     writer.writerow([epoch+1,index,attacker_index,list(selected_clients), list(idxs_users)])



        elif  args.method == 'FL_CPE':
                print("This is FL_CPE  selection ATTACK")
                # N = args.num_clients   # Number of base station antennas  --num_users
                N = 100
                M = args.perround   # Number of base station antennas  --num_users
                K = args.nworker  # Total number of clients
                P = args.p  # Total transmit power
                B = args.B  # Total transmit bandwidth
                index = args.conspirator_idx
                attacker_index =  args.attacker_index

                # Generate channel matrix with random complex entries
                H = np.random.randn(K, N) + 1j * np.random.randn(K, N)

                # Compute the total transmit power across all antennas
                total_power = np.sum(np.abs(H)**2)

                # Normalize the channel matrix to satisfy the power constraint
                H = H * np.sqrt(P / total_power)


                # def Update_CSI_CPE(H,K,N,attacker_id,conspirator_id):
                #Predict the clients based on Plain text
                selected_clients = Plain_Select(H,K,N)
                print("Original selected_clients is:",selected_clients[:M])

                # H_new  =  Update_CSI_CPE(H,K,N,selected_clients[15],attacker_index,1.2,0.2)

                H_new  =  Update_CSI_CPE(H,K,N,selected_clients[index],attacker_index,1.2,0.2)

                # backdoor_attacker.append(selected_clients[index])
                backdoor_attacker = []
                backdoor_attacker.append(selected_clients[index])

                # print(victim_idx,j_minus_1_idx,selected_clients[15])
                # print("The conspirator is:",selected_clients[conspirator])
                choices = Plain_Select(H_new,K,N)
                choices = choices[:M]
                print("conspirator is:",selected_clients[index])
                print("Attack clients by FL_CPE are:",choices)

                # Save original selected_clients[:10] and attack clients idxs_users[:10] to a CSV file
                # with open('./results/selected_CPE_clients_' + args.attack + '_' + args.agg + '_' + args.dataset + '_' + str(len(mal_index)) + '_' + str(time.strftime("%Y%m%d"))+ '.csv', 'a+', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     # writer.writerow(['Original Client ID', 'Attack Client ID'])
                #     writer.writerow([iter+1,selected_clients[index],attacker_index,list(selected_clients[:M]), list(idxs_users)])

        else:
            # The normal selection process
            # choices = np.random.choice(args.nworker, args.perround, replace=False)

            # The selection process with fix-attackers
            # Randomly select args.perround - 1 additional workers
            # Combine the fixed worker and the randomly chosen workers
            
            choices = mal_index.copy()  # Start with mal_index
            available_workers = [i for i in range(args.nworker) if i not in mal_index]
            choices.extend(np.random.choice(available_workers, args.perround - len(mal_index), replace=False))
            
            # Print the selected worker indices for this round
            print("Selected workers:", choices)
       
        
        with open('./results/selected_clients_' + args.attack + '_' + args.agg + '_' + args.method + '_' + args.dataset + '_' + str(len(mal_index)) + '_' + str(time.strftime("%Y%m%d"))+ '.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
                # writer.writerow(['Original Client ID', 'Attack Client ID'])
            writer.writerow(choices)


        if args.attack == 'modelpoisoning':
            dynamic_mal_index = np.random.choice(choices, args.malnum, replace=False)

        # copy network parameters
        params_copy = []
        for p in list(network.parameters()):
            params_copy.append(p.clone())

        for c in tqdm(choices):
            if args.attack == 'modelpoisoning' and c in dynamic_mal_index:
                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)

                for _ in range(0, args.localiter):
                    params_temp = []
                    for p in list(network.parameters()):
                        params_temp.append(p.clone())
                    delta_mal = mal_single(mal_train_loaders, train_loaders[c], network, criterion, optimizer, params_temp, device, [], round_idx, dist=True, mal_boost=2.0, path=args.agg)
                    for idx, p in enumerate(local_grads[c]):
                        local_grads[c][idx] = p + delta_mal[idx]
            
            elif c in mal_index and args.attack == 'backdoor':

                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)

                for _ in range(0, args.localiter):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                        attack_target = torch.zeros(args.batchsize, dtype=torch.long).to(device)
                        optimizer.zero_grad()
                        output = network(attack_feature)
                        loss = criterion(output, attack_target)
                        loss.backward()
                        optimizer.step()

                for idx, p in enumerate(network.parameters()):
                    local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

            else:
                for _ in range(0, args.localiter):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        feature = feature.to(device)
                        target = target.type(torch.long).to(device)
                        optimizer.zero_grad()
                        output = network(feature)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                if args.agg == 'iclr2022_bucketing' or args.agg == 'icml2021_history':
                    for idx, p in enumerate(network.parameters()):
                        local_grads[c][idx] = (1 - args.beta) * (params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()) + args.beta * local_grads[c][idx]
                else:
                    for idx, p in enumerate(network.parameters()):
                        local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

            # manually restore the parameters of the global network
            with torch.no_grad():
                for idx, p in enumerate(list(network.parameters())):
                    p.copy_(params_copy[idx])

        if args.attack == 'modelpoisoning':
            average_grad = []

            for p in list(network.parameters()):
                average_grad.append(np.zeros(p.data.shape))

            for c in choices:
                if c not in mal_index:
                    for idx, p in enumerate(average_grad):
                        average_grad[idx] = p + local_grads[c][idx] / args.perround

            np.save('./checkpoints/' + args.agg + 'ben_delta_t%s.npy' % round_idx, average_grad)

        # TODO: choices are not passed in as arguments, will the following two attacks deal with sub-sampled FL correctly?
        elif args.attack == 'trimmedmean':
            print('attack trimmedmean')
            local_grads = attack_trimmedmean(network, local_grads, mal_index, b=1.5)

        elif args.attack == 'krum':
            print('attack krum')
            for idx, _ in enumerate(local_grads[0]):
                local_grads = attack_krum(network, local_grads, mal_index, idx)

        elif args.attack == 'xie':
            print('attack Xie et al.')
            local_grads = attack_xie(local_grads, 1, choices, mal_index)

        # aggregation
        average_grad = []
        for p in list(network.parameters()):
            average_grad.append(np.zeros(p.data.shape))
        if args.agg == 'average':
            print('agg: average')
            s = time.time()
            for idx, p in enumerate(average_grad):
                avg_local = []
                for c in choices:
                    avg_local.append(local_grads[c][idx])
                avg_local = np.array(avg_local)
                average_grad[idx] = np.average(avg_local, axis=0)
            # print('average running time: ', time.time()-s)
        elif args.agg == 'krum':
            print('agg: krum')
            s = time.time()
            for idx, p in enumerate(average_grad):
                krum_local = []
                for c in choices:
                    krum_local.append(local_grads[c][idx])
                average_grad[idx], _ = krum(krum_local, f=args.malnum)
            # print('krum running time: ', time.time()-s)
        elif args.agg == 'filterl2':
            print('agg: filterl2')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                filterl2_local = []
                for c in choices:
                    filterl2_local.append(local_grads[c][idx])
                average_grad[idx] = filterL2(filterl2_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
            # print('filterl2 running time: ', time.time()-s)
        elif args.agg == 'mom_filterl2':
            print('agg: mom_filterl2')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                filterl2_local = []
                for c in choices:
                    filterl2_local.append(local_grads[c][idx])
                average_grad[idx] = mom_filterL2(filterl2_local, eps=args.malnum*1./args.nworker, sigma=args.sigma, delta=np.exp(-50+args.malnum))
            # print('mom_filterl2 running time: ', time.time()-s)
        elif args.agg == 'median':
            print('agg: median')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                median_local = []
                for c in choices:
                    median_local.append(local_grads[c][idx])
                average_grad[idx] = median(median_local)
            # print('median running time: ', time.time()-s)
        elif args.agg == 'trimmedmean':
            print('agg: trimmedmean')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                trimmedmean_local = []
                for c in choices:
                    trimmedmean_local.append(local_grads[c][idx])
                average_grad[idx] = trimmed_mean(trimmedmean_local)
            # print('trimmedmean running time: ', time.time()-s)
        elif args.agg == 'bulyankrum':
            print('agg: bulyankrum')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for c in choices:
                    bulyan_local.append(local_grads[c][idx])
                average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='krum')
            # print('bulyankrum running time: ', time.time()-s)
        elif args.agg == 'bulyanmedian':
            print('agg: bulyanmedian')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for c in choices:
                    bulyan_local.append(local_grads[c][idx])
                average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='median')
            # print('bulyanmedian running time: ', time.time()-s)
        elif args.agg == 'bulyantrimmedmean':
            print('agg: bulyantrimmedmean')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for c in choices:
                    bulyan_local.append(local_grads[c][idx])
                average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='trimmedmean')
            # print('bulyantrimmedmean running time: ', time.time()-s)
        elif args.agg == 'ex_noregret':
            print('agg: explicit non-regret')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                ex_noregret_local = []
                for c in choices:
                    ex_noregret_local.append(local_grads[c][idx])
                average_grad[idx] = ex_noregret(ex_noregret_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
            # print('ex_noregret running time: ', time.time()-s)
        elif args.agg == 'mom_ex_noregret':
            print('agg: mom explicit non-regret')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                ex_noregret_local = []
                for c in choices:
                    ex_noregret_local.append(local_grads[c][idx])
                average_grad[idx] = mom_ex_noregret(ex_noregret_local, eps=args.malnum*1./args.nworker, sigma=args.sigma, delta=np.exp(-50+args.malnum))
            # print('mom_ex_noregret running time: ', time.time()-s)
        elif args.agg == 'iclr2022_bucketing':
            print('agg: BYZANTINE-ROBUST LEARNING ON HETEROGENEOUS DATASETS VIA BUCKETING ICLR 2022')
            s = time.time()
            if prev_average_grad is None:
                prev_average_grad = []
                for p in list(network.parameters()):
                    prev_average_grad.append(np.zeros(p.data.shape))
                    shuffled_choices = np.random.shuffle(choices)
            bucket_average_grads = []
            for bidx in range(args.buckets):
                bucket_average_grads.append([])
                for idx, _ in enumerate(average_grad):
                    avg_local = []
                    for c in choices[bidx: bidx+args.perround//args.buckets]:
                        avg_local.append(local_grads[c][idx])
                    avg_local = np.array(avg_local)
                    bucket_average_grads[bidx].append(np.average(avg_local, axis=0))
            for bidx in range(args.buckets):
                norm = 0.
                for idx, _ in enumerate(average_grad):
                    norm += np.linalg.norm(bucket_average_grads[bidx][idx] - prev_average_grad[idx])**2
                norm = np.sqrt(norm)
                for idx, _ in enumerate(average_grad):
                    bucket_average_grads[bidx][idx] = (bucket_average_grads[bidx][idx] - prev_average_grad[idx]) * min(1, args.tau/norm)
            for idx, _ in enumerate(average_grad):
                avg_local = []
                for bidx in range(args.buckets):
                    avg_local.append(bucket_average_grads[bidx][idx])
                avg_local = np.array(avg_local)
                average_grad[idx] = np.average(avg_local, axis=0)
            prev_average_grad = copy.deepcopy(average_grad)
            # print('iclr2022_bucketing running time: ', time.time()-s)
        elif args.agg == 'icml2021_history':
            print('agg: Learning from History for Byzantine Robust Optimization ICML 2021')
            s = time.time()
            if prev_average_grad is None:
                prev_average_grad = []
                for p in list(network.parameters()):
                    prev_average_grad.append(np.zeros(p.data.shape))
            for c in choices:
                norm = 0.
                for idx, _ in enumerate(average_grad):
                    norm += np.linalg.norm(local_grads[c][idx] - prev_average_grad[idx])**2
                norm = np.sqrt(norm)
                for idx, _ in enumerate(average_grad):
                    local_grads[c][idx] = (local_grads[c][idx] - prev_average_grad[idx]) * min(1, args.tau/norm)
            for idx, _ in enumerate(average_grad):
                avg_local = []
                for c in choices:
                    avg_local.append(local_grads[c][idx])
                avg_local = np.array(avg_local)
                average_grad[idx] = np.average(avg_local, axis=0)
            prev_average_grad = copy.deepcopy(average_grad)
            # print('icml2021_history running time: ', time.time()-s)
        elif args.agg == 'clustering':
            print('agg: Secure Byzantine-Robust Distributed Learning via Clustering')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                avg_local = []
                for c in choices:
                    avg_local.append(local_grads[c][idx])
                average_grad[idx] = mom_krum(avg_local, f=args.malnum)
            # print('clustering running time: ', time.time()-s)


        params = list(network.parameters())
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.from_numpy(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)


        if (round_idx + 1) % args.checkpoint == 0:

            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)

            if args.attack == 'modelpoisoning':

                mal_test_loss = 0
                correct = 0
                with torch.no_grad():
                    for idx, (feature, mal_data, true_label, target) in enumerate(mal_train_loaders, 0):
                        feature = feature.to(device)
                        target = target.type(torch.long).to(device)
                        output = network(feature)
                        mal_test_loss += F.nll_loss(output, target, reduction='sum').item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                mal_test_loss /= len(mal_train_loaders.dataset)
                txt_file.write('%d, \t%f, \t%f, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy, mal_test_loss, 100. * correct / len(mal_train_loaders.dataset)))
                print('\nMalicious set: Accuracy: {:.4f}, Attack Success Rate: {}/{} ({:.0f}%)\n'.format(accuracy, correct, len(mal_train_loaders.dataset), 100. * correct / len(mal_train_loaders.dataset)))
            

            elif args.attack == 'backdoor':

                mal_test_loss = 0
                correct = 0
                with torch.no_grad():
                    for feature, target in test_loader:
                        feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                        target = torch.zeros(1, dtype=torch.long).to(device)
                        output = network(feature)
                        mal_test_loss += F.nll_loss(output, target, size_average=False).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                test_loss /= len(test_loader.dataset)
                txt_file.write('%d, \t%f, \t%f, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy, mal_test_loss, 100. * correct / len(test_loader.dataset)))
                print('\nMalicious set: Accuracy: {:.4f}, Attack Success Rate: {}/{} ({:.0f}%)\n'.format(accuracy, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


            else:

                txt_file.write('%d, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy))
                print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))