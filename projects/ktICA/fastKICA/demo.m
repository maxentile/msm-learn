% demo file of how to use fastkica

%
% Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton 
%


% parameters for fastkica
maxiter = 20;   % maximum number of iterations
sigma = 1.0;    % width of Gaussian kernel
thresh = 1e-6;  % convergence threshold (difference of HSIC values)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  LOAD DATA

% audio processing
% load data and randomly select 1500 (successive) samples
S = zeros(50000,3);
[S(:,1), Fs, bits] = wavread('source4.wav');
[S(:,2), Fs, bits] = wavread('source2.wav');
[S(:,3), Fs, bits] = wavread('source3.wav');
S = S';
idx = round(10000*(1+rand));
S = S(:,idx:idx+1499);

[m n] = size(S);    % m dimensions (number of signals); n sample size

% mixing matrix MM
MM = rand(m);
% mixed sources MS
MS = MM * S;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DEMIX

% initial demixing matrix X (should be orthogonal, i.e. X * X' = I)
% here: random matrix
% you may also use the result of Jade or FastICA for initialization
X = rand(m);
[X0 R] = qr(X);

% X0' * MS are estimated sources
% 
% Xout:     demixing matrix at final iteration: estimated sources are
%           Xout'*MS
% XS:       X for each iteration
% hsics:    HSIC value at each iteration
[Xout, XS, hsics] = fastkica(MS, X0, maxiter, sigma, thresh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  PLOT result

figure;
subplot(1,2,1);
maxind = find(hsics > 0, 1, 'last');
plot(1:maxind, hsics(1:maxind));
axis tight;
xlabel('iteration');
ylabel('HSIC');

subplot(1,2,2);
amari_errors = zeros(maxind+1, 1);
amari_errors(1) = amariD(X0'*MM);       % initial error
for i=1:maxind
    amari_errors(i+1) = amariD(XS(:,:,i)'*MM);
end
plot(0:maxind, amari_errors);
xlabel('iteration');
ylabel('Amari distance');
axis([1 maxind 0 max(amari_errors)]);
