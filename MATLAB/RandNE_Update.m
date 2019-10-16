function U_list = RandNE_Update(A, delta_A, U_list, Ortho, seed)
% Inputs:
%   A: original adjacency matrix
%   delta_A: the changes of adjacency matrix
%   U_list: original decomposed parts
%   Ortho: whether use orthogonal projection
%   seed: random seed
% Outputs:
%   U_list: updated decomposed parts

q = size(U_list, 1);                % order + 1
[N,d] = size(U_list{1});            % dimension
N_new = size(delta_A, 1);        
delta_U = cell(q, 1);

if N_new > N                   % if new users, adjust dimensionality
    [temp_row, temp_col, temp_v] = find(A);
    A = sparse(temp_row, temp_col, temp_v, N_new, N_new);
    if nargin < 5
        error('Not enough configurations for new users!');
    end
    rng(seed);
    U_list{1} = [U_list{1}; normrnd(0,1/sqrt(d),N_new - N,d)];
    if Ortho == 1                           
        U_list{1} = GS(U_list{1});
    end
    for i = 2:q
        U_list{i} = [U_list{i};zeros(N_new - N,d)];
    end
    N = N_new;
end

delta_U{1} = sparse(N, d);     % calculate changed parts
for i = 2:q
    delta_U{i} = delta_A * U_list{i-1} + A * delta_U{i-1} + delta_A * delta_U{i-1};
end

for i = 2:q
    U_list{i} = U_list{i} + delta_U{i};
end

end