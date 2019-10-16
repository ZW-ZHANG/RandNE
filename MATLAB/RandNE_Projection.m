function U_list = RandNE_Projection(A,q,d,Ortho,seed)
% Inputs:
%   A: sparse adjacency matrix
%   q: order
%   d: dimensionality
%   Ortho: whether use orthogonal projection
%   seed: random seed
% Outputs:
%   U_list: a list of R, A * R, A^2 * R ... A^q * R

N = size(A,1);

rng(seed);                               % set random seed
U_list = cell(q + 1,1);                       % store each decomposed part
U_list{1} = normrnd(0,1/sqrt(d),N,d);         % Gaussian random matrix
if Ortho == 1                            % whether use orthogonal projection
    U_list{1} = GS(U_list{1});
end
for i = 2: (q + 1)                       % iterative random projection
    U_list{i} = A * U_list{i-1};
end
end