function P_ortho = GS(P)
% Input:
%   P: n x d random matrix
% Output:
%   P_ortho: each column orthogonal while maintaining length
% Performing modified Gram¨CSchmidt process

[~,d] = size(P);
temp_l = zeros(d,1);
for i = 1:d
    temp_l(i) = sqrt( sum(P(:,i) .^2) );
end
for i = 1:d
    temp_row = P(:,i);
    for j = 1:i-1
        temp_j =  P(:,j);
        temp_product = temp_j' * temp_row  / temp_l(j)^2;
        temp_row = temp_row - temp_product * temp_j ; 
    end
    temp_row = temp_row * (temp_l(i) / sqrt(temp_row' * temp_row));
    P(:,i) = temp_row;
end
P_ortho = P;
end