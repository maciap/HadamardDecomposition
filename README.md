# HadamardDecomposition


Alternating gradient descent for: 
 - Hadamard decomposition
 - Mixed Hadamard decomposition 
 
 
Example usage

```python
from data_utils import makerealdata_full_rank_gaussian
from alternating_gradient_descent import scaled_alternating_gradient_descent_hadDec
D = makerealdata_full_rank_gaussian(250,250,0,1) 
D_estimate,  [D_1_estimate, D_2_estimate], [A_1, B_1, A_2, B_2], [all_diffs1, all_diffs2] , terminated =  scaled_alternating_gradient_descent_hadDec(D, 6, 0.01, 100000)

print(f"Approximation error: {all_diffs2[1]}")
```
Example usage with real data 
```python
import cv2 
dataset_name = "rgb_dog"
img = cv2.imread("data/" + dataset_name +".jpg") 
D = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
D_estimate,  [D_1_estimate, D_2_estimate], [A_1, B_1, A_2, B_2], [all_diffs1, all_diffs2] , terminated =  scaled_alternating_gradient_descent_hadDec(D, 20, 0.01, 100000)

plt.figure(figsize=(12, 6)) 
plt.imshow(D_estimate, cmap='gray'), 
plt.title(f'# vectors = {comps[i]*2}', fontsize=19) 
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.tight_layout() 
plt.savefig("output/rgb_dog_reconstruction_example.pdf") 
plt.show()
```

![My Image](output/rgb_dog_reconstruction_example.pdf)




The code is tested in Python 3.8.18 and Ubuntu 20.04.2 LTS. 


Requirements: 
  - numpy 
  - numba 
  
 
 
  
