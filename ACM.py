import numpy as np


def align_in_window(V, l, r, V_thr=10):
    """Aligning the set of Vm for each neuron in network by first spike in the time window [l, r]. Vm is a membrane potential of a neuron.

    Parameters:
    V (array): Array of shape (T, N) where T is a length of Vm timeseries and N is a number of neurons
    l (int): left border of the time window
    r (int): right border of the time window
    V_thr (float): the value of membrane potential that defines the threshold for spike 

    Returns:
    tuple: (V_new (array): membrane potentials that aligned by first spike, times_to_first_spike (list): times for each timeseries of Vm from begining to first spike)

   """
    
    V = V[l:r, :]
    length = r - l
    spikes = np.logical_and(V[1:-length // 2 - 1, :] < V_thr, V[2:length//2, :] > V_thr)
    ts = [np.where(spikes[:,i])[0] for i in range(V.shape[1])]
    times_to_first_spike = [t[0] if len(t) > 0 else 0 for t in ts]
    new_len = r - l - max(times_to_first_spike)
    V_new = np.array([V[times_to_first_spike[i]:times_to_first_spike[i] + new_len, i] for i in range(V.shape[1])]).T
    return V_new, times_to_first_spike


def calc_chi2(Vm):
    """Calculation of Chi^2 metric of synchronization. 

    Parameters:
    Vm (array): Array of shape (T, N) where T is a length of Vm timeseries and N is a number of neurons

    Returns: 
    float: Chi^2 metric
    
   """

    V = np.mean(Vm, axis=1)
    sigma_V = np.std(V)
    sigma_Vi = np.std(Vm, axis=0)
    return sigma_V ** 2 / np.mean(sigma_Vi ** 2)


def calc_R2(Vm):
    """Calculation of R^2 metric of synchronization in ACM approach. 

    Parameters:
    Vm (array): Array of shape (T, N) where T is a length of Vm timeseries and N is a number of neurons

    Returns: 
    float: R^2 metric
    
   """    
    V_new, _ = align_in_window(Vm, 0, len(Vm))
    return calc_chi2(V_new)


def calc_L(Vm):
    """Calculation of L as additional metric of synchronization for ACM approach. 

    Parameters:
    Vm (array): Array of shape (T, N) where T is a length of Vm timeseries and N is a number of neurons

    Returns: 
    int: number of clusters of synchronization
    
   """    
    
    _, times_first_spikes = align_in_window(Vm, 0, len(Vm))
    return len(np.unique(times_first_spikes))


def calc_L_accurate(Vm):
    """Calculation of L as additional metric of synchronization for ACM approach. The difference here that this function makes calculations more accurate.

    Parameters:
    Vm (array): Array of shape (T, N) where T is a length of Vm timeseries and N is a number of neurons

    Returns: 
    int: number of clusters of synchronization
    
   """        
    _, times_first_spikes = align_in_window(Vm, len(Vm) - 100000, len(Vm))
    L_old = np.unique(times_first_spikes)
    
    L_cur = np.sort(L_old)
    L_indexes = np.arange(len(L_cur))[np.argsort(L_old)]
    Ls_diff = np.diff(L_cur)
    local_max = (Ls_diff[:-2] < Ls_diff[1:-1]) & (Ls_diff[1:-1] > Ls_diff[2:]) & (Ls_diff[1:-1] > np.quantile(np.unique(Ls_diff[1:-1]), 0.75))
    L_acc = np.sum(local_max) + 1
    
    local_max_indexes, = np.where(local_max)
    color = np.zeros_like(L_cur)
    color[L_indexes[:local_max_indexes[0]]] = 0
    if len(local_max_indexes) > 1:
        for i in range(1, len(local_max_indexes)):
            color[L_indexes[local_max_indexes[i - 1]:local_max_indexes[i]]] = i
    color[L_indexes[local_max_indexes[-1]:]] = np.max(color) + 1

    groups_lens = [len(color[color == i]) for i in np.unique(color)]
    L_lg = sum(groups_lens > np.max(groups_lens) - 100)
    
    if len(L_old) > 100:
        return len(L_old), L_lg, color
    
    return L_acc