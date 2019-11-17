%%% The toy example in section 4.2
%%% To reproduce the results, you need to run this script 3 times 
%%% by setting index = 1,2,3 respectively

mag = 1000; % 1: small mass fail; 1000: large mass fail
u = @(x) mag*(-x).*(x<0)+mag*x.*(x>=0); % potential energy
u_grad = @(x) -mag*(x<0)+mag*(x>=0); % gradient of potential energy
u_par = @(x) exp(-u(x)); % partition function
inter = 0.2; % bin width
end1 = -3; % plot range left end
end2 = 0; % plot range right end
xs1 = end1:inter:end2; % bins
Z = sum(u_par(xs1)); % total partition function
p = 0; % initial momentum
q = 0.1; % initial position
b = 500; % total number of cycles
step = 0.03; % step size
L = 5; % number of steps in each path
qs = []; % collection of positions
m1 = 1; % one mass
m2 = 1e-2; % another mass
meanss = []; % average of samples
run = 20; % monte carlo times

%index = {1,2,3}
index = 1;
color = ['g','b','r'];
pos_shift = [-5,0,5]; % avoid lines overlapping with each other
prob = [0,0.5,1.0]; % prob = 0, m=m1 (HMC); prob = 1, m=m2 (HMC); prob=0.5, QHMC


for j=1:run
    mean0 = 1;
    means = [];
    % HMC/QHMC
    for i=1:b
        if rand>prob(index)
            m = m1;
        else
            m = m2;
        end
        p = normrnd(0,sqrt(m));
        [qf,pf] = leap_frog(u_grad,m,q,p,step,L);
        if mh(u,m,q,qf,p,pf)==1
            qs = [qs; q];
            q = qf;
        end
        mean0 = (mean0 * (i-1) + q)/i;
        means = [means;mean0];
    end
    meanss = [meanss;means'];
end

%%% plot
x = 1:20:b
mean1 = mean(meanss,1);
std1 = std(meanss,1);
e = errorbar(x+pos_shift(index),mean1(x),std1(x),'.');
e.Color = color(index);
hold on 
xlim([-10,460])
if index==3
    plot(1:b,zeros(1,b),'k--');
end



% [contents, bins] = hist(qs,xs);
% hold on
% acc_rate = length(qs)/b
% p_eff = sum(qs>end1 & qs<end2);
% p_tot = length(qs);
% %ylim([0,0.8])
% %bar(bins, contents*p_tot/p_eff)
% bar(bins(2:end-1), contents(2:end-1)/p_eff)
% plot(xs,u_par(xs)/Z*1,'--','LineWidth',2)
% xlabel('position','FontSize',14)
% ylabel('probability density','FontSize',14)
% %title("f(x)=20|x|^{1/10}, m\sim exp(N("+num2str(log_m0)+","+num2str(sigma_m)+"^2))")
% title("Explicit Adaptation",'FontSize',10)
        

