% A sample run on the BlogCatalog Dataset
% load data
load('BlogCatalog');

% Parameters
q = 2;
d = 128;
Ortho = 1;
seed = 0;
weights = [1,0.1,0.001];

% embedding for adjacency matrix
U_list = RandNE_Projection(A,q,d,Ortho,seed);
U = RandNE_Combine(U_list,weights);
% reconstruction
prec = Precision_Np(A,sparse(length(A),length(A)),U,U,1e6);
semilogx(1:1e6,prec);

% embedding for transition matrix
A_tran = spdiags(1 ./ sum(A,2),0,length(A),length(A));
U_list = RandNE_Projection(A_tran,q,d,Ortho,seed);
U = RandNE_Combine(U_list,weights);