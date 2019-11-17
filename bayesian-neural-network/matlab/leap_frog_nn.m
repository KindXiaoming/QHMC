function [qf,pf,zetaf] = leap_frog_nn(n,therm_m, u_grad,m,q,p,zeta,step,L)
p = p - step * u_grad/2;
for i=1:L
    q = q + m^(-1) * step * p;
    if i~=L
        p = p - step * u_grad;
    end
    zeta = zeta + step * 1/therm_m*(sum(p.^2)/m-n);
end
p = p - step * u_grad/2;
qf = q;
pf = p;
zetaf = zeta;
end
