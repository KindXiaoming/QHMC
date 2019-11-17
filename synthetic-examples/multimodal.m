%%% we implement HMC/QHMC for a toy multimodal distribution (section 6.1.3)
%%% double well distribution x^4-4x^2 %%%%%

u = @(x) x.^4-4*x.^2;  % potential energy
u_grad = @(x) 4*x.^3-8*x;  % gradient of potential energy
u_par = @(x) exp(-u(x)); % partition function
inter = 0.1; % bin width
end1 = -2; % plot range, left end
end2 = 2; % plot range, right end
xs = end1:inter:end2; % bins
Z = sum(u_par(xs)); % total partition function
q = sqrt(2); % initial position (right minimum)
p = 0; % initial momentum
b = 5000; % total number of cycles
step = 0.03; % step size
L = 5; % number of steps in each simulation path
qs = []; % collection of positions

% mass in QHMC
log_m0 =  1; % \mu_m in QHMC
sigma_m = 0; % \sigma_m in QHMC. If set 0 -> HMC.

% QHMC/HMC
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
       

%%% plot
[contents, bins] = hist(qs,xs);
hold on
acc_rate = length(qs)/b
p_eff = sum(qs>end1 & qs<end2);
p_tot = length(qs);
%ylim([0,0.8])
%bar(bins, contents*p_tot/p_eff)
bar(bins(2:end-1), contents(2:end-1)/p_eff)
plot(xs,u_par(xs)/Z*1,'--','LineWidth',2)
        
        

