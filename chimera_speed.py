import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import jit, numpy as jnp, ops, lax, vmap


V_THRESH = 10
KERNEL_SIZE = 10000


@jit
def shift_on_velocity(V: jnp.ndarray, velocity: float):
    return vmap(jnp.roll)(V, (velocity * jnp.arange(V.shape[0])).astype(jnp.int32))


@jit
def get_freqs(V: jnp.ndarray, V_thresh: float, step=.1):
    spikes = (V[1:-1, :] < V_thresh) & (V[2:, :] > V_thresh)
    return spikes.mean(axis=0) * 1000 / step


@jit
def freq_std_for_velocity(V: jnp.ndarray, velocity: float, thresh: float, power: float):
    V_shifted = shift_on_velocity(V, velocity)
    freqs = get_freqs(V_shifted, thresh)
    return freqs.std()


@jit
def max_similar_freq_for_velocity(V: jnp.ndarray, V_thresh: float, velocity: float, allowed_diff: float):
    def body(i, state):
        mins, maxs, rolled, best_ans = state
        
        maxs = jnp.maximum(maxs, rolled)
        mins = jnp.minimum(mins, rolled)
        rolled = jnp.roll(rolled, 1)
        
        best_ans = jnp.where(jnp.min(maxs - mins) < allowed_diff, i + 1, best_ans)
        
        return mins, maxs, rolled, best_ans
    
    freqs = get_freqs_for_velocity(V, V_thresh, velocity)
    min_init = jnp.full(len(freqs), jnp.inf)
    max_init = jnp.full(len(freqs), -jnp.inf)

    _, _, _, best_ans = jax.lax.fori_loop(0, len(freqs), body, (min_init, max_init, freqs, 0))
    
    return best_ans


@jit
def get_freqs_for_velocity(V: jnp.ndarray, V_thresh: float, velocity: float):
    return get_freqs(shift_on_velocity(V, velocity), V_thresh)


@jit
def min_minimal_freq_for_velocity(V: jnp.ndarray, V_thresh: float, velocity: float):
    f = get_freqs_for_velocity(V, V_thresh, velocity)
    return (f <= 2).sum()


jitted_minimal_freq = jit(vmap(min_minimal_freq_for_velocity, in_axes=(None, None, 0)))
jitted_similar = jit(vmap(max_similar_freq_for_velocity, in_axes=(None, None, 0, None)))
jitted_std = jit(vmap(freq_std_for_velocity, in_axes=(None, 0, None, None)))


@jit
def max_combined_for_velocities(V: jnp.ndarray, V_thresh: float, velocities: float, allowed_diff: float, ALPHA: float, POWER: float):
    similarity = jitted_similar(V, V_thresh, velocities, allowed_diff)
    similarity /= similarity.max() - similarity.min()
    
    freq_min_elem = jitted_minimal_freq(V, V_thresh, velocities) 
    diff = freq_min_elem.max() - freq_min_elem.min()
    freq_min_elem /= jnp.where(diff == 0, 1, diff)
    
    return ALPHA * freq_min_elem + (1 - ALPHA) * similarity


def find_best_vel(V, func='combined', KERNEL_SIZE=20000, V_THRESH=10, ALLOWED_DIFF=0.5, ALPHA=0.5, POWER=2):
    
    velocities = jnp.linspace(0, 0.1, 2000)
    
    if func == 'similarity':
        similars = jitted_similar(V[-KERNEL_SIZE:], V_THRESH, velocities, ALLOWED_DIFF)
    
    if func == 'minimal':
        similars = jitted_minimal_freq(V[-KERNEL_SIZE:], V_THRESH, velocities)

    if func == 'std':
        similars = jitted_std(V[-KERNEL_SIZE:], velocities, V_THRESH, POWER)

    if func == 'combined':
        similars = max_combined_for_velocities(V[-KERNEL_SIZE:], V_THRESH, velocities, ALLOWED_DIFF, ALPHA, POWER)
        
    similars_cur = similars.copy()
    velocities_cur = velocities.copy()
    best_vel = velocities_cur[jnp.argmax(similars_cur)]
    f = dynamic_chimera_freq(V, best_vel, KERNEL_SIZE, V_THRESH)
    f_old = get_freqs(V[-KERNEL_SIZE:], V_THRESH)
    
    print(best_vel)
    
    return best_vel.item(), f


def find_best_vel_cond(V, func='combined', KERNEL_SIZE=20000, V_THRESH=10, ALLOWED_DIFF=10., ALPHA=0.5, POWER=4):    
    velocities = jnp.linspace(0, 0.1, 2000)
    
    if func == 'similarity':
        similars = jitted_similar(V[-KERNEL_SIZE:], V_THRESH, velocities, ALLOWED_DIFF)
        plt.figure(figsize=(7, 3))
        plt.title('similar')
        plt.plot(velocities, similars)
        plt.show()
    
    if func == 'minimal':
        similars = jitted_minimal_freq(V[-KERNEL_SIZE:], V_THRESH, velocities)
        plt.figure(figsize=(7, 3))
        plt.title('min f')
        plt.plot(velocities, similars)
        plt.show()

    if func == 'std':
        similars = jitted_std(V[-KERNEL_SIZE:], velocities, V_THRESH, POWER)
        plt.figure(figsize=(7, 3))
        plt.title('std')
        plt.plot(velocities, similars)
        plt.show()

    if func == 'combined':
        similars = max_combined_for_velocities(V[-KERNEL_SIZE:], V_THRESH, velocities, ALLOWED_DIFF, ALPHA, POWER)
        plt.figure(figsize=(7, 3))
        plt.title('combined')
        plt.plot(velocities, similars)
        plt.show()
        
    similars_cur = similars.copy()
    velocities_cur = velocities.copy()
    best_vel = velocities_cur[jnp.argmax(similars_cur)]
    f = dynamic_chimera_freq(V, best_vel, KERNEL_SIZE, V_THRESH)
    f_old = get_freqs(V[-KERNEL_SIZE:], V_THRESH)
    
    print(best_vel, jnp.argmax(similars_cur))
    
    i = 0
    while i < 50:
        if np.min(f_old) <= (np.min(f) - 2) or np.max(f) > 15:
            if jnp.argmax(similars_cur) <= 50:
                break
            similars_cur = similars_cur[:jnp.argmax(similars_cur) - 50]
            velocities_cur = velocities_cur[:len(similars_cur)]
            best_vel = velocities_cur[jnp.argmax(similars_cur)]
            f = dynamic_chimera_freq(V, best_vel, KERNEL_SIZE, V_THRESH)
            i += 1
        else:
            plt.figure(figsize=(7, 3))
            plt.scatter(np.arange(500), f)
            plt.xlabel('# neurons')
            plt.ylabel('f')
            plt.ylim(0, 20)
            plt.show()
            
            plt.figure(figsize=(7, 3))
            plt.plot(velocities, similars)
            plt.title(func)
            plt.show()
            
            return best_vel.item(), f
    return -1, f


def dynamic_chimera_freq(V, velocity, kernel_size, V_thresh, draw=False):
    V_tn = V[-kernel_size:]    
    V_shifted = shift_on_velocity(V_tn, velocity)        
    f = get_freqs(V_shifted, V_thresh)
    if draw:
        plt.figure(figsize=(7, 3))
        plt.scatter(np.arange(500), f)
        plt.xlabel('# neurons')
        plt.ylabel('f')
        plt.show()
    return f
