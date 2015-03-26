function [X, XS, hsics] = fastkica(MS, Xin, maxiter, sigma, thresh)
%
% INPUT:
% MS:       mixed signals, m x n (m sources, n samples)
% Xin:      initial demixing matrix: Xin' * MS are estimated sources
%           (initial guess)
% maxiter:  maximum number of iterations
% sigma:    width of Gaussian kernel
% thresh:   convergence threshold: stops if difference in subsequent values
%           of HSIC is less than thresh
%
% OUTPUT:
% X:        demixing matrix: X' * MX are estimated sources
% XS:       sequence of X for each iteration
% hsics:    HSIC at each iteration
%
% The algorithm terminates if either the difference in hsic values is less
% than 'thresh' or the maximum number of iterations (maxiter) is reached.
%
% Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton
%

if ~exist('hessChol.m','file')
    addpath('utils');
end

[m n] = size(MS);         % m signals, n samples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  PREPROCESSING: whitening
%   multiply mixed signal MS by a whitening matrix WW such that WW*MM has
%   an identity covariance matrix

% center signal
shift = mean(MS, 2);                 
MS = MS - shift * ones (1, n);

% whiten signal, WS is whitened signal
[Q R] = qr(MS',0);
[theta lamda v] = svd(R');
WW = sqrt(n)*diag(diag(lamda).^(-1))*theta';
WS = WW * MS;

% correct Xin for whitening and make orthogonal
% then X' * WS = Xin' * MS
[U,S,V] = svd(Xin'*inv(WW));
X = U*V';
X = X';  % we use the transpose

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

XS = zeros(m,m,maxiter);  % stores X at each iteration
hsics = zeros(1,maxiter); % stores HSIC after each iteration

n2 = n*n;

sigmas = ones(1,m)*sigma; % kernel width
etas = ones(1,m) * 1e-6;  % Cholesky precision
maxdim = 60;              % maximum number of columns for the Cholesky factors

for iter = 1 : maxiter

    % Cholesky
    ks = {}; %kernel halves
    inds = {};
    Y = X' * WS; % estimated sources
    for iS=1:m
        [G,Pvec] =chol_gauss(Y(iS,:), sigmas(iS), etas(iS)*n);
        ddim = size(G,2);
        inds{iS} = Pvec(1:ddim)+1;
        % This is a restriction to save memory, remove/adapt as you wish.
        % ddim is the d dimension of the decomposition matrices (they are n
        % x d, from incomplete Cholesky). It is here restricted to be at
        % most maxdim=60. 
        if ddim>maxdim
            inds{iS} = Pvec(1:maxdim)+1;
            fprintf('%d: ddim too large ', iS);
        end            
        [a,Pvec]=sort(Pvec); %new pvec contains indices of old pvec
        ks{iS} = G(Pvec,:);
    end

    % HSIC of current estimate
    hsics(iter) = hsicChol(ks,n,m);
    
    % Euclidean gradient: entry-wisely computed
    [G] = compDerivChol(WS, X, inds, sigmas);
    G = G/n2;

    %     compute approximate Hessian
    HS = hessChol(WS,X,ks, sigma);
    
    EG = G * X; % same as (-X'*EG)'
    RG = (EG - EG') / 2;                            % Gradient in parameter space
    X = X * expm( RG.* HS );                        % map Newton direction onto SO(m)
    XS(:,:,iter) = WW'*X;                           
    % unmixed data is X' * WW * MS = XS(:,:,iter)' * MS

    % check for convergence
    if iter>1 && (abs(hsics(iter)-hsics(iter-1))<thresh)
        X = XS(:,:,iter);
        return;
    end
end
X = XS(:,:,maxiter);
