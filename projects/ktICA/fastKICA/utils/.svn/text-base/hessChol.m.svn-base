function [HS] = hessChol(WS,X,ks, varargin)
% function [HS] = hessChol(WS,X,ks, [sigma])
%
% Hessian 
% WS:       whitened signal
% X :       current point (demixing matrix)
% HS:       approximate Hessian (with Cholesky)
% sigma:    width of Gaussian kernel (here we assume that all kernels are
%           of the same width), default: sigma = 1;

%
% Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton 
%

[m n] = size(WS);
n2 = n^2;
if nargin < 4
    sigma = 1;
else
    sigma = varargin{1};
end
eta = 1e-8 * n;
Y = X' * WS;
mt = zeros(3,m); % three "magic terms"

% Magic Terms, see AISTATS paper
for nos = 1:m

    G = ks{nos};

    % M1
    v = ones(1,n)*G;   % 1 x d
    mt(1,nos) = v*v'/n2;

    % M3
    v2 = (Y(nos,:).^2 * G) * (G' * ones(n,1)); % 1 x d
    mt(3,nos) = v2/n2;

    % M2
    v = Y(nos,:)*G; % d x 1
    mt(2,nos) = (v*v')/n2;
end

% Hessian
HS = zeros(m);
t2 = 2/(sigma^2);
t4 = 4/(sigma^4);
for row = 1 : m
    for col = (row+1) : m
        HS(row,col) = 1 / (t2* mt(1,row)*mt(2,col) + t2*mt(2,row)*mt(1,col) + ...
            t4 * mt(2,row) * mt(2, col) - t4*mt(3,row)*mt(3,col));
    end
end
HS = HS + HS';
