# ACM
This code calculates adaptive coherence measure (ACM) and chimera speed: https://www.sciencedirect.com/science/article/pii/S096007792100895X

We propose a robust universal approach to identify multiple network dynamical states, including stationary and travelling chimera states based on an adaptive coherence measure. 
		Our approach allows automatic disambiguation of synchronized clusters, travelling waves, chimera states, and asynchronous regimes. In addition, our method can determine the number of clusters in the case of cluster synchronization. We further couple our approach with a new speed calculation method for travelling chimeras. We validate our approach by an example of a ring network of type II Morris-Lecar neurons with asymmetrical nonlocal inhibitory connections where we identify a rich repertoire of coherent and wave states. We propose that the method is robust for the networks of phase oscillators and extends to a general class of relaxation oscillator networks.
    
   To estimate coherence of a network, one can use $\chi^2$-parameter [1]:
	

$$\chi^2 = \frac{\sigma^2_V}{\frac{1}{N}\sum^{N}_{i=1}{\sigma^2_{V_i}}}$$,

where $V(t)$ is the time series of membrane potentials in the representative time window and $\sigma^2_V$ is variance of average membrane potential of the network $V(t) = \frac{1}{N}\sum_{i=1}^{N}V_i(t)$:

$$\sigma^2_V = \langle V^2(t)\rangle_t - \langle V(t)\rangle_t^2$$,

and $\sigma^2_{V_i}$ is variance of membrane potential of the i-th neuron:

$$\sigma^2_{V_i} = \langle V_i^2(t)\rangle_t - \langle V_i(t)\rangle_t^2$$.
	
	The parameter $\chi^2$ takes the values in [0,1]. For asynchronous state $\chi^2 \approx 1 / \sqrt{N}$ \cite{golomb2001mechanisms}, but for macroscopic neural networks $\chi^2 \rightarrow 0$. In case of full synchrony $\chi^2 = 1$ and for the values $0 < \chi^2 < 1$ the chimera states may exist. 




1. Golomb, D., Hansel, D., & Mato, G. (2001). Mechanisms of synchrony of neural activity in large networks. In Handbook of biological physics (Vol. 4, pp. 887-968). North-Holland.
