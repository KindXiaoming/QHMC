%%% just used to monitor the sampling process and tune paramters
diabetes = importdata('diabetes.txt');
data = diabetes.data;
n = 442; % number of samples
para = 10; % number of parameters
X = data(:,1:10);
y = data(:,11);
y = y - mean(y);
y = y/norm(y);
X_mean = sum(X,1)/n;
for i=1:10
    X(:,i) = X(:,i)-X_mean(i);
end

X_norm = vecnorm(X);
for i=1:10
    X(:,i) = X(:,i)/X_norm(i);
end

%X = rand(n,para); % Input matrix. Each row: one sample; Each column: one feature
beta0 = randn(para,1);
mu = 1;
p0 = 1;
eps = 5e-2;
pnormp = @(x) (sum(abs(x).^p0));
lambda = 0
pnorm_grad = @(x) p0./(abs(x).^(1-p0)+eps).*(x>0)+p0./(abs(x).^(1-p0)+eps)*(-1).*(x<=0);
u = @(beta) mu*1/(2*n)*norm(y-X*beta)^2 + lambda * pnormp(beta);
u_grad = @(beta) mu*(-X'*y + X'*X*beta) + lambda*pnorm_grad(beta);

q = beta0;
p = 0;
L = 5;
qs = [];
step = 0.3;
b = 2000; % total number of cycles
b0 = 1000; % burn in
losses = [];
step2 = 0.005;

for i=1:b0
    q = q - step2*u_grad(q);
    qs = [qs; q'];
    losses = [losses; u(q)];
end

for i=1:b
    logm = normrnd(log_m0,sigma_m);
    m = 10^logm;
    p = normrnd(0,sqrt(m));      
    [qf,pf] = leap_frog(u_grad,m,q,p,step,L);
    if mh(u,m,q,qf,p,pf)==1
        q = qf;
        qs = [qs; q'];
    end
    loss = u(q);
    losses = [losses; loss];
end

figure
acc_rate = (length(qs)-b0)/b
hold on
for i=1:10
    subplot(2,6,i)
    histogram(qs(b0:end,i))
end

subplot(2,6,11)
plot(1:(b+b0), losses)
