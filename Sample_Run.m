% A sample run on the BlogCatalog Dataset
% load data
load('BlogCatalog');

% Common parameters
d = 128;
Ortho = 1;
seed = 0;

% embedding for adjacency matrix for reconstruction
q = 2;
weights = [1,0.1,0.001];
U_list = RandNE_Projection(A,q,d,Ortho,seed);
U = RandNE_Combine(U_list,weights);
prec = Precision_Np(A,sparse(length(A),length(A)),U,U,1e6);
semilogx(1:1e6,prec);

% embedding for transition matrix for classification
q = 3;
weights = [1,1e2,1e4,1e5];
A_tran = spdiags(1 ./ sum(A,2),0,length(A),length(A)) * A;
U_list = RandNE_Projection(A_tran,q,d,Ortho,seed);
U = RandNE_Combine(U_list,weights);
