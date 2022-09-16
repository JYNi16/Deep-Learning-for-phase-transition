PCA for 2DIsing model 

Requirements: scikit-learn, numpy, numba 

Quick start:

1. python MC_sample.py to generate the datasets.
   
   Datasets contain three parts： 20_L, 40_L and 80_L, in which "L" represent the size of model.    
   
   For each parts, we generate the datas in the range of temperature from 1 to 3.0. 
   
   For each temperature point, we generate 100 datas.

2. python sklearn_pca.py.