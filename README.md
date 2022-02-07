# Differentiable implementation of Matrix Exponential with Pade Approximation
This is an implementation of the algorithm that computes matrix exponential of matrices using its [13/13] Pade approximant as in [The Scaling and Squaring Method for the Matrix Exponential Revisited](https://dl.acm.org/doi/10.1137/04061101X).

The code supports matrices <img src="https://render.githubusercontent.com/render/math?math=\in \mathbb{C}^{n \times n}">

Specifically we implement Algorithm 2.3 of the paper.

A minimal example is as follows:

```python
import torch
from expm_basic import MatrixExp(A_r, A_c)
## Let input matrix A have real part A_r and imaginary part A_c.
A_r  = torch.randn(10,10)
A_c = torch.randn(10,10)
Out_r, Out_c = MatrixExp(A_r, A_c)
```


The code is an adaptation of the implementation of the `expm` function in Scipy.

# Notes
* U and V from lines 18 and 19 in the Algorithm is given by the function *pade13_scaled*. Put A = A_r + iA_c in the equations of U and V to get the implementation in the code.
* The preprocessing steps have been omitted. Will be added in the future.



