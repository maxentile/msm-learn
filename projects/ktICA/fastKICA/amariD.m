function Perf=amariD(Per);
% Amari distance - distance between two matrices. Beware: it does not verify the axioms
%                  of a distance. It is always between 0 and 1.
% This code is based on Francis Bach's amari_distance.m

m=size(Per,1);
Perf=[sum((sum(abs(Per))./max(abs(Per))-1)/(m-1))/m; sum((sum(abs(Per'))./max(abs(Per'))-1)/(m-1))/m];
Perf=mean(Perf);
