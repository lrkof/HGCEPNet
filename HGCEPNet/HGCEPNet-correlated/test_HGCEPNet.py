import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import argparse
import numpy as np
from data import *
from sp_model_k import *
from ChebNet_modify import *
from helper import *
from collections import defaultdict
import time

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
# MIMO data parameters
parser.add_argument('--k', type=int, default=3, help='QAM constellation, k=1 for 4QAM, 2 for 16QAM, 3 for 64QAM')
parser.add_argument('--iter', type=int, default=100, help='number of iteration/replication')
parser.add_argument('--batch_size', type=int, default=1000, help='number of samples per setting')
parser.add_argument('--save_result', action='store_true', default=False, help='save result in a file')
parser.add_argument('--n_hid', type=int, default=64, help='')
parser.add_argument('--Su', type=int, default=8, help='')
parser.add_argument('--Nr', type=int, default=16, help='Number of receiver')
parser.add_argument('--snrdb_low', type=int, default=25, help='lower bound of snrdb training')
parser.add_argument('--snrdb_high', type=int, default=50, help='lower bound of snrdb training')
parser.add_argument('--ill', type=int, default=1, help='1 for ill-conditioned; 0 for random')
parser.add_argument('--test_rho_list', type=float, default=np.array([0.6]), help='test rho list')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning_rate')
parser.add_argument('--epoch', type=int, default=850, help='total number of epoches used in training')
parser.add_argument('--cheb_order', type=int, default=3, help='order of chebshev')
parser.add_argument('--seed', type=int, default=3, help='random seed')
args = parser.parse_args()

DIR = '../correlation_results_rep'
if not os.path.exists(DIR):
    os.makedirs(DIR)


def generate_data(batch_size):
    # data_dict = {int(NT): {} for NT in NT_list}
    data_dict = {int(NT): {rho: defaultdict(float) for rho in args.test_rho_list} for NT in NT_list}
    for Nt in NT_list:
        for rho in args.test_rho_list:
            for snrdb in snrdb_list[Nt]:
                snr = 10**(snrdb/10.0)
                H, y, x, snr, indices, noise_sigma = generate_MIMO_data_batch_ori(
                    Nt, Nr, args.k, batch_size, snr, snr, rho, ill)
                L, y_hat, alpha = sp_GNN_data(H, y)
                noise_sigma2 = noise_sigma**2
                all_ones = torch.ones(y.shape)
                H_t_e = torch.matmul(H.permute(0, 2, 1), all_ones)
                feature = torch.cat((y_hat, alpha * noise_sigma.unsqueeze(-1) * H_t_e), dim=2)
                feature = feature.to(dtype).to(device)
                label = indices.to(device)
                L = L.to(dtype).to(device)
                y = y.squeeze().to(dtype).to(device)
                H = H.to(dtype).to(device)
                noise_sigma2 = noise_sigma2.to(dtype).to(device)
                data_dict[int(Nt)][rho][snrdb] = (H, y, L, feature, noise_sigma2, label)
    return data_dict


def eval_model(model, H, y, L, feature, noise_sigma2, label, Nt):
    batch_size = L.shape[0]
    with torch.no_grad():
        last_time = time.time()
        out = ChebNet.forward(model, H, y, noise_sigma2, feature, L, args.k)
        now_time = time.time()
        time_elapsed = now_time - last_time
        y_pred_soft = soft_max(out)
        loss = 0.0
        x_hats = []
        for idx_user in range(label.shape[-1]):
            loss_each = criterion(out[:,idx_user,:], label[:,idx_user]) 
            loss = loss + loss_each
            x_hat = np.matmul(y_pred_soft[:, idx_user, :].to('cpu').detach().numpy(), constellation_expanded)
            x_hats.append(x_hat)
        x_hats = np.concatenate(x_hats, 1)
        x_indices = find_index(torch.tensor(x_hats), constellation)
        # acc = (out.argmax(dim=2) == label).cpu().numpy().sum()/(2 * Nt * batch_size)
        acc = (x_indices == label.cpu()).sum()/(x_indices.numel())
    return acc.item(), loss.item(), time_elapsed


torch.manual_seed(args.seed + 12345)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)

Nt = args.Nr; Nr = args.Nr
ill = True if args.ill == 1 else False
soft_max = nn.Softmax(dim=2)
constellation = generate_QAM_signal_list(args.k)
constellation_expanded = np.expand_dims(constellation, axis=1)
n_class = len(constellation)
cond = 'ill' if ill else 'well'

BASE_DIR = '../tested_HGCEPNet_correlated_lr{}_spk_v2'.format(args.learning_rate)

n_feature = 4
cheb_order = args.cheb_order
EP_iter = 10; GNN_iter = 2
beta = 0.7
ChebNet = ChebNet(EP_iter, beta, args.n_hid, args.Su, constellation, device, dtype)
model = sp_GNN(cheb_order, n_class, GNN_iter, args.n_hid, args.Su, n_feature).to(device)

model_name = '{}x{}_k{}_{}_sp{}'.format(
    Nr, Nr, args.k, cond, cheb_order)

checkpoint = torch.load('{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}_best_mix.pt'.format(
    BASE_DIR, model_name, args.epoch, args.snrdb_low, args.snrdb_high))

model.load_state_dict(checkpoint['model_state_dict'])
ChebNet.load_state_dict(checkpoint['chebnet_state_dict'])
criterion = nn.CrossEntropyLoss().to(device=device)
NT_list = np.asarray([int(Nt),])
# for each NT, generate data with different snr
snrdb_list = {int(Nt): np.arange(args.snrdb_low, args.snrdb_high + 1)}
print(args)
result_dict = {int(NT): {rho: defaultdict(float) for rho in args.test_rho_list} for NT in NT_list}
time_dict = {float(rho): {int(snr): 0 for snr in snrdb_list[Nt]} for rho in args.test_rho_list}
for it in range(args.iter):
    print(f'{it}-th Test Data is Generating..........')
    data_dict = generate_data(args.batch_size)
    for Nt in NT_list:
        for rho in args.test_rho_list:
            for snr in snrdb_list[Nt]:
                print("{} - {} - Nr: {} | Nt: {} |rho: {} | snr: {}".format(it, "spGNN", Nr, Nt, rho, snr))
                # select batch data according to NT and snr
                H, y, L, feature, noise_sigma2, label = data_dict[int(Nt)][rho][snr]
                acc, loss, time_elapsed = eval_model(model, H, y, L, feature, noise_sigma2, label, Nt)
                result_dict[Nt][rho][snr] += acc
                time_dict[rho][snr] += time_elapsed

for NT in NT_list:
    for rho in args.test_rho_list:
        for snr in snrdb_list[NT]:
            result_dict[NT][rho][snr] /= args.iter

for rho in args.test_rho_list:
    for snr in snrdb_list[int(Nt)]:
        time_dict[rho][snr] /= args.iter


filename = '{}/spGNNkv2_{}_ep{}_lr{}_dbmin{}_dbmax{}_seed{}_mix_rho=[0.6]'.format(
    DIR, model_name, args.epoch, args.learning_rate, args.snrdb_low, args.snrdb_high, args.seed)
torch.save({'res': result_dict, 'time': time_dict}, filename)

print('----------------------Test for HGCEPNet Correlated Done----------------------')
print('Test results are save at {}'.format(filename))
