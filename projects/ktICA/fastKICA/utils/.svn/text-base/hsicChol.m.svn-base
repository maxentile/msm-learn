function [outScore]=hsicChol(ks,N,m)
%
% function [outScore]=hsicChol(ks,N,m)
%
% HSIC of the estimate represented by ks
% ks:   array, where ks{i} is an N x d_i matrix R such that RR' is the kernel matrix for
%       source i
% N:    number of samples
% m:    number of sources


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dimZ=[0];
allZMatrices=[];

% centre matrices
for i=1:m
    G=centerpartial(ks{i}); 
    %note: R=G(Pvec,:) is defined such
    %that RR'=K (kernel matrix). This is
    %then fed into centering program
    %size of G is N * d_i, where d_i << N
    allZMatrices=[allZMatrices G];
    dimZ=[dimZ;size(G,2)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%construct the contrast matrix 
dimZcumulative=cumsum(dimZ);

outScore = 0;
for k=1:m-1
    for l=k+1:m
        Zk = allZMatrices(:,dimZcumulative(k)+1:dimZcumulative(k+1));
        Zl = allZMatrices(:,dimZcumulative(l)+1:dimZcumulative(l+1));
        outScore = outScore + trace ( (Zl'*Zk)*(Zk'*Zl) ); 
    end
end
outScore = outScore*1/N^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G2=centerpartial(G1)
% CENTERPARTIAL - Center a gram matrix of the form K=G*G'

[N,NG]=size(G1);
G2 = G1 - repmat(mean(G1,1),N,1);
