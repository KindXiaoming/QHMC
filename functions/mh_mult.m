function [accept,flag] = mh(u,m,q_orig,q_prop,p_orig,p_prop)
% mh decides if accepting the proposal
% Need to consider kinetic term
a = cellfun(@(x) sum(x.^2),p_prop,'un',0);
P2_prop = sum([a{:}]);
H_prop = u(q_prop{:}) + P2_prop/(2*m);
b = cellfun(@(x) sum(x.^2),p_orig,'un',0);
P2_orig = sum([b{:}]);
[b{:}]/m*0.0001
H_orig = u(q_orig{:}) + P2_orig/(2*m); 
flag = isnan(-H_prop+H_orig)|isinf(-H_prop+H_orig);
a = rand();
accept = a < min(1,exp(-H_prop+H_orig));
%a
%P2_prop
%P2_orig
disp(H_prop-H_orig);
%exp(-H_prop+H_orig)
end
