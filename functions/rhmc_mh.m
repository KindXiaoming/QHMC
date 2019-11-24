function accept = rhmc_mh(u,invm,q_orig,q_prop,p_orig,p_prop)
% mh decides if accepting the proposal
% Need to consider kinetic term
H_prop = u(q_prop) + 0.5*transpose(p_prop)*invm*p_prop;
%size(p_prop)
H_orig = u(q_orig) + 0.5*transpose(p_orig)*invm*p_orig; 
accept = rand < min(1,exp(-H_prop+H_orig));
end

