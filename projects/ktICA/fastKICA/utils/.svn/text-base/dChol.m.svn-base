function [derivKL] = dChol(Knd,Kdd,dKnd, Lnd, Ldd, dLnd, indsK, indsL)
%
% Matlab interface for pairwise derivative for kernels K and L
% Knd:          n x d submatrix of K
% Kdd:          inverted d x d submatrix of K
% dKnd:         d(Knd)
% Lnd:          n x d submatrix of K
% Ldd:          inverted d x d submatrix of K
% dLnd:         d(Knd)
% indsK, indsL: indices from incomplete Cholesky decomposition
%
% Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton
%


[n,dl] = size(Lnd);
dk = length(indsK);
[ndk,m] = size(dKnd);

derivKL = zeros(2,m);

% column-centered Lnd
HL = Lnd - repmat(mean(Lnd),n,1);

% derivative of d x d submatrix of K and L
indsext = repmat(indsK,1,dk) + kron(0:n:(n*dk-1),ones(1,dk));
dKdd = dKnd(indsext,:);
indsext = repmat(indsL,1,dl) + kron(0:n:(n*dl-1),ones(1,dl));
dLdd = dLnd(indsext,:);
clear indsext;

% derivative wrt row (of X) for K
derivKL(1,:) = dChol2(HL, Ldd, Knd, Kdd, dKnd, dKdd);

% column-centered Knd
HK = Knd - repmat(mean(Knd),n,1);
% derivative wrt row (of X) for L
derivKL(2,:) = dChol2(HK, Kdd, Lnd, Ldd, dLnd, dLdd);
