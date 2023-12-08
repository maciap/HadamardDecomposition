# HadamardDecomposition


Alternating gradient descent for: 
 - Hadamard decomposition
 - Mixed Hadamard decomposition 
 
 
Example usage

```python
from data_utils import makerealdata_full_rank_gaussian
from alternating_gradient_descent import scaled_alternating_gradient_descent_hadDec
D = makerealdata_full_rank_gaussian(250,250,0,1) 
D_estimate,  [D_1_estimate, D_2_estimate], [A_1, B_1, A_2, B_2], [all_diffs1, all_diffs2] , terminated =  scaled_alternating_gradient_descent_hadDec(D, total_rank, eta =  0.01, T = 100000)

print(f"Approximation error: {all_diffs2[1]}")
```


The code is tested in Python 3.8.18 and Ubuntu 20.04.2 LTS. 


Requirements: 
  - numpy 
  - numba 
  
 
 
  
