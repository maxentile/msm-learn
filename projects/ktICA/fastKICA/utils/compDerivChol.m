function [gradJ] = compDerivChol(WS, X, inds, sigmas)
% function [gradJ] = compDerivChol(WS, X, inds, sigmas)
% 
% gradient (including Cholesky factorization, see the Book Chapter in "Large Scale Kernel Machines")
%
% WS:       whitened signals
% X:        demixing matrix
% inds:     array of cholesky indices: inds{i} is index vector for kernel i
% sigmas:   m x 1 vector: kernel width for each component

% First the necessary submatrices are constructed. The function dChol
% performs the multiplication to get the gradient

%
% Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton 
%

N=size(WS,2);       % number of data points
m=size(WS,1);       % number of components

ridge = 1e-6;

if ispc
    Y = X'*WS;
    X = X';

    % parts of Cholesky gradient
    kern_nd = {};   % n x d submatrices
    kern_dd = {};   % d x d submatrices for inversion
    dkern_nd = {};  % derivative of n x d submatrix

    for i=1:m
        kern_nd{i} = getKern(Y(i,:), Y(i,inds{i}), sigmas(i));
        kern_dd{i} = kern_nd{i}(inds{i},:);
        kern_dd{i} = inv(kern_dd{i} + ridge * eye(length(inds{i})));
        dkern_nd{i} = dKmn(kern_nd{i}, inds{i}, X(i,:), WS, sigmas(i));
    end
    gradJ = zeros(m,m);

    % sum up pairwise gradients
    for i=1:(m-1)
        for j=(i+1):m
            g = dChol(kern_nd{i},kern_dd{i},dkern_nd{i}, kern_nd{j}, kern_dd{j}, dkern_nd{j}, inds{i}, inds{j});
            gradJ([i,j],:) = gradJ([i,j],:) + g;
        end
    end
    clear kern_nd;
    clear kern_dd;
    clear dkern_nd;
else
    Y = X'*WS;
    X = X';

    % parts of Cholesky gradient
    kern_nd = {};
    kern_dd = {};
    dkern_nd = {};

    for i=1:m
        kern_nd{i} = getKern(Y(i,:), Y(i,inds{i}), sigmas(i));
        kern_dd{i} = kern_nd{i}(inds{i},:);
        kern_dd{i} = inv(kern_dd{i} + ridge * eye(length(inds{i})));
        dkern_nd{i} = dKmnLin(kern_nd{i}, inds{i}, X(i,:), WS, sigmas(i));
    end
    gradJ = zeros(m,m);

    % sum up pairwise gradients
    for i=1:(m-1)
        for j=(i+1):m
            g = dCholLin(kern_nd{i},kern_dd{i},dkern_nd{i}, kern_nd{j}, kern_dd{j}, dkern_nd{j}, inds{i}, inds{j});
            gradJ([i,j],:) = gradJ([i,j],:) + g;
        end
    end
    clear kern_nd;
    clear kern_dd;
    clear dkern_nd;
end
