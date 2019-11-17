%%% investigate the dependence of \mu_m and \sigma_m on
%%% acceptance rate and distribution accuracy with lp examples
%%% (not presented in paper)

function [mse,acc_rate] = lp_test(log_m0,sigma_m)
p0= 0.2+rand()*3.8;
a = 10^(rand()*2-1);
u = @(x) a*x.^p0.*(x>0)+a*(-x).^p0.*(x<=0);
eps = 1e-3;
u_grad = @(x) a*p0/(x.^(1-p0)+eps).*(x>0)+a*p0/((-x).^(1-p0)+eps).*(x<=0);
u_par = @(x) exp(-u(x));
inter = 1;
end1 = -10;
end2 = 10;
xs = end1:inter:end2;
Z = sum(u_par(xs));
q = 0.1;
p = 0;
b = 5000; % total number of cycles
step = 0.03;
L = 5;
qs = [];

for i=1:b
    logm = normrnd(log_m0,sigma_m);
    m = 10^logm;
    p = normrnd(0,sqrt(m));
    [qf,pf] = leap_frog(u_grad,m,q,p,step,L);
    if mh(u,m,q,qf,p,pf)==1
        qs = [qs; q];
        q = qf;
    end
end
       
[contents, bins] = hist(qs,xs);
acc_rate = length(qs)/b;
test = contents;
if sum(contents)==0
    test = 0;
else
    test = contents/sum(contents);
end
truth = u_par(xs)/Z;
mse = sum((truth-test).^2);


