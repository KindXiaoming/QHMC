%%% section 6.1.1 lp example, and section 6.1.2 example

% section 6.1.2 example
% x1 = 3;
% x2 = 1;
% h1 = 0;
% h2 = 2;
% a = 4*h2/x2^2;
%u = @(x) (-x-x1).*(x<-x1)-h1*(-x1<=x & x<0)+a*x.*(x-x2).*(0<=x & x<x2)+(x-x2).*(x>=x2);
%u_grad = @(x) -1*(x<-x1)+a*(2*x-x2).*(0<=x & x<x2)+1*(x>=x2);

% section 6.1.1 example f(x)=a|x|^p
a = 1; % magnitude
p0 = 1; % p in Lp (not to confuse with p, the momentum)
u = @(x) a*x.^p0.*(x>0)+a*(-x).^p0.*(x<=0); % potential
eps = 1e-3; % small number, avoid numerical divergent gradients in lp
u_grad = @(x) a*p0/(x.^(1-p0)+eps).*(x>0)+a*p0/((-x).^(1-p0)+eps).*(x<=0); % gradient of potential
u_par = @(x) exp(-u(x)); % partition function
inter = 0.2; % bin width
end1 = -5; % plot range, left end
end2 = 5; % plot range, right end
xs = end1:inter:end2; % number of bins
Z = sum(u_par(xs)); % partition function
q = 0.1; % initial position
p = 0; % initial momentum
b = 20000; % total number of cycles
step = 0.1; % step size
L = 5; % number of steps in each simulation path
qs = []; % collection of position

% mass in QHMC
log_m0 = 2; % \mu_m in QHMC
sigma_m = 1; % \sigma_m in QHMC

% HMC/QHMC
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

% plot
figure('Renderer', 'painters', 'Position', [10 10 400 300])
% plot(xs,u(xs))

[contents, bins] = hist(qs,xs);
hold on
acc_rate = length(qs)/b
p_eff = sum(qs>end1 & qs<end2);
p_tot = length(qs);
%ylim([0,0.8])
%bar(bins, contents*p_tot/p_eff)
bar(bins(2:end-1), contents(2:end-1)/p_eff)
plot(xs,u_par(xs)/Z*1,'--','LineWidth',2)
xlabel('x','FontSize',23)
ylabel('probability density','FontSize',23)
%title("f(x)=20|x|^{1/10}, m\sim exp(N("+num2str(log_m0)+","+num2str(sigma_m)+"^2))")
title("m\sim 10^{N("+num2str(log_m0)+","+num2str(sigma_m)+"^2)}",'FontSize',25)
        

