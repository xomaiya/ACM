# ACM
This code calculates adaptive coherence measure (ACM) and chimera speed: https://www.sciencedirect.com/science/article/pii/S096007792100895X

We propose a robust universal approach to identify multiple network dynamical states, including stationary and travelling chimera states based on an adaptive coherence measure. Our approach allows automatic disambiguation of synchronized clusters, travelling waves, chimera states, and asynchronous regimes. In addition, our method can determine the number of clusters in the case of cluster synchronization. 

The criterion is based on the $\chi^2$-parameter [1] and involves the optimization problem:

$$R^2 = \max_{\Delta\mathbf{t} = (\Delta t_1, \Delta t_2, .., \Delta t_N)} \chi^2(\{V_i(t - \Delta t_i)\}_{i=1}^N)$$,

where $\Delta\mathbf{t} = (\Delta t_1, \Delta t_2, .., \Delta t_N)$ is a time delay vector that contains $L$ unique time lags, and $\chi^2$-defines as follow:

$ \chi^2 = \frac{\sigma^2_V}{\frac{1}{N}\sum^{N}_{i=1}{\sigma^2_{V_i}}}$,

where $\sigma^2_V$ is variance of average membrane potential of the network $$V(t) = \frac{1}{N}\sum_{i=1}^{N}V_i(t)$$, and $#\sigma^2_{V_i}$# is variance of membrane potential of the i-th neuron. 

Using both criteria (the number of unique time lags $L$ and the value of $R^2$) one can easily identify a dynamical regime. If $R^2$ is close to zero, an asynchronous state is observed. For a chimera state, its value ranges from zero to one: $0<R^2<1$. In the other cases of global synchronization, states consisting of only synchronous clusters and travelling waves, $R^2$ is close to one, and to classify the states we need to use the number of unique delays $L$. For travelling waves $L$ is equal to $N /k$ ($k$ is a number of waves in the ring), for clustered state it is between $1$ and $N$, for global synchronization it equals $1$.      
    
  


1. Golomb, D., Hansel, D., & Mato, G. (2001). Mechanisms of synchrony of neural activity in large networks. In Handbook of biological physics (Vol. 4, pp. 887-968). North-Holland.
