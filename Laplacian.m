function L = Laplacian(X)
% Calculate the Laplacian Matrix of X
%
%   X�� NxD array
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 20;
    options.WeightMode = 'HeatKernel';
    options.t = 1;

    S = constructW(X, options) ;  
    L = diag(sum(S, 2))- S; 
end

