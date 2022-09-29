AutoEncoders for 2DIsing model 

Requirements: Pytorch, numpy

Quick start:

python train.py 

1. python MC_sample.py to generate the datasets.

   Here, we generate the datas with size of 45 in the range of temperature from 1 to 3.0. 
   
   For each temperature point, we generate 500 datas.

2. model.file. 
   
   Neural Netword based AutoEncoders.

3. python train.py.

   training file. 

4. python infer.py 

   Inferece using saved model.  