function accept = mh(u,m,q_orig,q_prop,p_orig,p_prop)
% mh decides if accepting the proposal
% Need to consider kinetic term
H_prop = u(q_prop) + sum(p_prop.^2)/(2*m);
H_orig = u(q_orig) + sum(p_orig.^2)/(2*m); 
accept = rand < min(1,exp(-H_prop+H_orig));
end

