import numpy as np
import scipy
import math
import torch
import torch.utils.data
import random
from scipy.linalg import sqrtm
from scipy.linalg import toeplitz

# generate noise following Gaussian distribution
def generate_noise(Nr, noise_sigma): # noise_sigma is the std for the complex noise
    v = np.random.normal(0, noise_sigma, 2*Nr)
    return v

# generate random channel and each element is sampled from i.i.d gaussian distribution
def generate_gaussian_channel(Nt, Nr, mu=0):
    sigma = 1.0/np.sqrt(2. * Nr)
    Hr = np.random.normal(mu, sigma, size=(Nr, Nt))
    Hi = np.random.normal(mu, sigma, size=(Nr, Nt))
    h1 = np.concatenate((Hr, -1.0 * Hi), axis = 1)
    h2 = np.concatenate((Hi, Hr), axis = 1)
    H = np.concatenate((h1, h2), axis = 0)   
    return H

def ill_helper(N, x):
    return toeplitz([x**i for i in range(N)])


def generate_ill_channel(Nt, Nr, rho=0.5):
    H = np.random.normal(0, 1.0/np.sqrt(2.0 * Nr), size=(2*Nr, 2*Nt))
    psi = ill_helper(2*Nr, rho)
    phi = ill_helper(2*Nt, rho)
    psi_2 = sqrtm(psi)
    phi_2 = sqrtm(phi)
    A = psi_2 @ H @ phi_2
    return A


def add_estimation_error(H, sigma_e):
    # H batch_size,2NR,2NT
    batch_size = H.shape[0]
    NR = NT = H.shape[2] // 2
    Er = torch.empty((batch_size, NR, NT)).normal_(mean=0, std=sigma_e / np.sqrt(2. * NR))
    Ei = torch.empty((batch_size, NR, NT)).normal_(mean=0, std=sigma_e / np.sqrt(2. * NR))
    E1 = torch.cat((Er, Ei), dim=2)
    E2 = torch.cat((-Ei, Er), dim=2)
    E = torch.cat((E1, E2), dim=1)
    H_hat = H - E

    return H_hat


# generate the valid signal values w.r.t different QAM constellation
# k = 1, 2, 3 correspond to QPSK (i.e. 4QAM), 16QAM, 64QAM
def generate_QAM_signal_list(k): # unnormalized
    vaild_signal_list = np.linspace(int(-(2**k -1)), int(2**k - 1), int(2**k))
    return vaild_signal_list


# generate x by randomly selecting signal values from valid signal list
def generate_rand_index(constellation, Nt):
    indice = [np.random.choice(range(len(constellation))) for i in range(2 * Nt)]
    # s = constellation[indice]
    return indice


# Nt = 3; Nr = 4; k = 2; batch_size = 5; snr_min = 1; snr_max = 5
# generate a batch of samples, Nr <= Nr
def generate_MIMO_data_batch_ori(Nt, Nr, k, batch_size, snr_min, snr_max, rho, ill=False):
    # 如过ill==1则为correlated 否则是 iid rayleigh
    if ill:
        channel_matrix = np.array([generate_ill_channel(Nt, Nr, rho) for i in range(batch_size)])
    else:
        channel_matrix = np.array([generate_gaussian_channel(Nt, Nr, mu=0) for i in range(batch_size)])
    snr = torch.empty((batch_size, 1)).uniform_(snr_min, snr_max)
    channel_sigma2 = 1.0 / Nr
    noise_sigma2 = 2.0 * Nt * (4**k - 1) * channel_sigma2 / (3.0 * snr) 
    noise_sigma2 = noise_sigma2 * 0.5
    noise_sigma = np.sqrt(noise_sigma2)
    noise = np.array([generate_noise(Nr, noise_sigma_i) for noise_sigma_i in noise_sigma])
    constellation = generate_QAM_signal_list(k)
    indices = np.array([generate_rand_index(constellation, Nt) for i in range(batch_size)])
    x = constellation[indices]
    y = np.array([channel_matrix[i] @ x[i] + noise[i] for i in range(batch_size)])
    x = torch.from_numpy(x).to(dtype=torch.float64)
    indices = torch.from_numpy(indices).to(dtype=torch.int64)
    y = torch.from_numpy(y).to(dtype=torch.float64)
    y = y.unsqueeze(-1); x = x.unsqueeze(-1)
    channel_matrix = torch.from_numpy(channel_matrix).to(dtype=torch.float64)
    return channel_matrix, y, x, snr, indices, noise_sigma


def generate_MIMO_data_batch_imcsi(Nt, Nr, k, batch_size, snr_min, snr_max, rho, sigma_e, ill=False):
    # 如过ill==1则为correlated 否则是 iid rayleigh
    if ill:
        channel_matrix = np.array([generate_ill_channel(Nt, Nr, rho) for i in range(batch_size)])
    else:
        channel_matrix = np.array([generate_gaussian_channel(Nt, Nr, mu=0) for i in range(batch_size)])
    snr = torch.empty((batch_size, 1)).uniform_(snr_min, snr_max)
    channel_sigma2 = 1.0 / Nr
    noise_sigma2 = 2.0 * Nt * (4 ** k - 1) * channel_sigma2 / (3.0 * snr)
    noise_sigma2 = noise_sigma2 * 0.5
    noise_sigma = np.sqrt(noise_sigma2)
    noise = np.array([generate_noise(Nr, noise_sigma_i) for noise_sigma_i in noise_sigma])
    constellation = generate_QAM_signal_list(k)
    indices = np.array([generate_rand_index(constellation, Nt) for i in range(batch_size)])
    x = constellation[indices]
    x = torch.from_numpy(x).to(dtype=torch.float64)
    indices = torch.from_numpy(indices).to(dtype=torch.int64)
    channel_matrix = torch.from_numpy(channel_matrix).to(dtype=torch.float64)
    H_E = add_estimation_error(channel_matrix, sigma_e)
    noise = torch.from_numpy(noise).to(dtype=torch.float64)
    y = torch.bmm(H_E, x.unsqueeze(-1)) + noise.unsqueeze(-1)
    x = x.unsqueeze(-1)
    # print(f'y.shape:{y.shape}')
    return H_E, y, x, snr, indices, noise_sigma


def generate_valid_data_batch(Nt, Nr, k, batch_size, snr_min, snr_max, rho_low, rho_high, ill=False):
    if ill:
        channel_matrix = np.array([generate_ill_channel(Nt, Nr, rho=np.random.uniform(rho_low, rho_high)) for i in range(batch_size)])
    else:
        channel_matrix = np.array([generate_gaussian_channel(Nt, Nr, mu=0) for i in range(batch_size)])
    snr = torch.empty((batch_size, 1)).uniform_(snr_min, snr_max)
    channel_sigma2 = 1.0 / Nr
    noise_sigma2 = 2.0 * Nt * (4**k - 1) * channel_sigma2 / (3.0 * snr)
    noise_sigma2 = noise_sigma2 * 0.5
    noise_sigma = np.sqrt(noise_sigma2)
    noise = np.array([generate_noise(Nr, noise_sigma_i) for noise_sigma_i in noise_sigma])
    constellation = generate_QAM_signal_list(k)
    indices = np.array([generate_rand_index(constellation, Nt) for i in range(batch_size)])
    x = constellation[indices]
    y = np.array([channel_matrix[i] @ x[i] + noise[i] for i in range(batch_size)])
    x = torch.from_numpy(x).to(dtype=torch.float64)
    indices = torch.from_numpy(indices).to(dtype=torch.int64)
    y = torch.from_numpy(y).to(dtype=torch.float64)
    y = y.unsqueeze(-1); x = x.unsqueeze(-1)
    channel_matrix = torch.from_numpy(channel_matrix).to(dtype=torch.float64)
    return channel_matrix, y, x, snr, indices, noise_sigma


# transform the data for the spectral method
def sp_GNN_data(channel_matrix, y):
    H = channel_matrix
    two_NT = int(H.shape[-1])
    batch_size = int(H.shape[0])
    H_t = H.permute(0,2,1)
    HTH = torch.matmul(H_t, H)
    lambda_max = [max(torch.linalg.eigh(HTH[i])[0]).item() for i in range(batch_size)]
    lambda_max = torch.tensor(np.array(lambda_max))
    # alpha = 1.0/lambda_max
    # add condition number
    cond_numbers = torch.linalg.cond(H)  
    alpha = 1.0 / (lambda_max + 0.1 * cond_numbers)  
    alpha = alpha.view(-1, 1, 1)
    y_hat = alpha * torch.matmul(H_t, y)

    scaled_HTH = alpha * HTH
    I = torch.eye(two_NT).unsqueeze(0).repeat(batch_size, 1, 1)
    L = I - scaled_HTH

    # add channel feature enhance
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    S_normalized = S / S.max(dim=1, keepdim=True)[0] 
    sv_feats = torch.stack([S_normalized.mean(dim=1),
                            S_normalized.std(dim=1)], dim=-1)  

    sv_feats = sv_feats.unsqueeze(1).expand(-1, two_NT, -1)
    y_hat = torch.cat([y_hat, sv_feats], dim=-1)

    # H_norm = H / torch.norm(H, dim=1, keepdim=True)  
    # W = alpha * torch.matmul(H_norm.permute(0, 2, 1), H_norm)
    # L = torch.eye(W.size(1)) - W

    return L, y_hat, alpha

