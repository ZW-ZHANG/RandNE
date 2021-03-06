# RandNE
This is the official implementation of "[Billion-scale Network Embedding with Iterative Random Projection](https://zw-zhang.github.io/files/2018_ICDM_RandNE.pdf)" (ICDM 2018).

We provide two implementations: MATLAB and Python. Note that the MATLAB version is used in producing original results in the paper.

### Requirements
```
MATLAB R2017a
or 
Python >= 3.5.2
numpy >= 1.14.2
scipy >= 1.1.0
pandas >= 0.22.0
sklearn >= 0.0
```

### Usage
##### Example Usage
```
See Sample_Run.m or Sample_Run.py for a sample run on the BlogCatalog network
```
##### Functions
```
Sample_Run: a sample run on the BlogCatalog network, see annotations for detail
RandNE_Projection: calculate projection results of different orders by performing iterative random projection
    Inputs: sparse adjacency matrix, order q, dimension d, whether use orthogonal projection, random seed
    Outputs: a cell containing decomposed parts U_0 ... U_q
RandNE_Combine: combine different orders with given weights
    Inputs: a list of decomposed parts generated by RandNE_Projection, a vector of weights for each order w_0 ... w_q
    Outputs: final embedding vectors U
RandNE_Update: update embeddings for dynamic networks
    Inputs: original adjacency matrix, the changes of adjacency matrix, original decomposed parts, whether use orthogonal projection, random seed
    Outputs: updated decomposed parts
GS: perform Gram–Schmidt process to obtain orthogonal projections
```
##### A Note on Hyper-parameters
```
We find empirically that our method is somewhat sensitive to hyper-parameters (adjacency matrix/transition matrix, order q and weights for different orders w). 
A recommended way for tuning hyper-parameters is to use grid search on cross-validation sets, which should be efficient since order and weights only affect the last step of our algorithm (RandNE_Combine).
The range for searching is suggested as follows: order is from 1 to 3 and weights can be searched by w_{i+1} = beta_i w_i where beta_i is from {0.01,0.1,1,10,100}.
A rule of thumb is that adjacency matrix should have decreasing weights (i.e. beta_i <= 1) and transition matrix should have increasing weights (i.e. beta_i >= 1).
The hyper-parameters used in our paper for BlogCatalog dataset is listed as a reference:
Reconstruction: adjacency matrix, q = 1, weights = [1, 0.1]
Link Prediction: adjaecncy matrix, q = 2, weights = [1, 1, 0.01]
Node Classification: transition matrix, q = 3, weights = [1, 100, 10000, 100000]
```
### Cite
If you find this code useful, please cite our paper:
```
@inproceedings{zhang2018billion,
  title={Billion-scale Network Embedding with Iterative Random Projection},
  author={Zhang, Ziwei and Cui, Peng and Li, Haoyang and Wang, Xiao and Zhu, Wenwu},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={787--796},
  year={2018},
  organization={IEEE}
}
```