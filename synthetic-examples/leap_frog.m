function [qf,pf] = leap_frog(u_grad,m,q,p,step,L)
p = p - step * u_grad(q)/2;
for i=1:L
    q = q + m^(-1) * step * p;
    if i~=L
        p = p - step * u_grad(q);
    end
end
p = p - step * u_grad(q)/2;
qf = q;
pf = p;
end

