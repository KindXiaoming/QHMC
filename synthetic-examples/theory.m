% In this script, we examine HMC (no momentum re-sampling) 
% for the |x|^p-type functions
m = 1; % mass
eps = 0.01; % step size
a = 1; % magnitude of function
k = 4; % power index
xc = (4*m/(eps^2*a*k))^(1/(k-2)); % critical radius (not presented in paper)
%xc = 1 % if k=2, choose a random xc is OK.
delta = 0; % perturb the particle away from xc
%x = xc-delta;
x = 0.54*xc;
p = 0; % initial momentum
u = @(x) a*x.^k.*(x>0)+a*(-x).^k.*(x<=0); % potential energy
u_grad = @(x) a*k*x.^(k-1).*(x>0)-a*k*(-x).^(k-1).*(x<=0); % gradient of potential
xs = []; % collect positions

% HMC
step = 100;
for i=1:step
    if i==1
        p = p - u_grad(x)*eps/2;
        continue
    end
    x = x + p/m*eps;
    p = p - u_grad(x)*eps;
    xs = [xs; x];
end

xs = [xc; xs];

%plot(1:step,ones(1,step)*xc,'r--')
hold on
%plot(1:step,-ones(1,step)*xc,'r--')
plot(2:step,xs(2:end))
ylim([-200,200])
%histogram(xs,-60:5:60,'Normalization','pdf')

% plot(3:2:step,xs(3:2:step),'g')
% hold on
% plot(2:2:step,xs(2:2:step),'g')
% leng = length(1:2:step);
% plot(1:2:step,ones(1,leng)*xc,'--r')
% plot(2:2:step,-ones(1,leng)*xc,'--r')
%ylim([-2,2])
        
