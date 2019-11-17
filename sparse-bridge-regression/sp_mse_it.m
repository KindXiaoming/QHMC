%%% section 6.2, (log) loss of testing mse
%%% to reproduce Figure 11 (a), one needs to run this script twice
%%% by setting sigma_m=0(HMC) and sigma_m>2(QHMC)

rng(1);

% read data and standarize data
diabetes = importdata('diabetes.txt');
data = diabetes.data;
n_train = 300; % number of samples
n_test = 142;
para = 10; % number of parameters
X = data(:,1:10);
y = data(:,11);
y = y - mean(y);
y = y/std(y);
X_mean = sum(X,1)/n;
for i=1:10
    X(:,i) = X(:,i)-X_mean(i);
end

X_norm = vecnorm(X);
for i=1:10
    X(:,i) = X(:,i)/X_norm(i)*sqrt(441);
end

% split data to train and test
X_train = X(1:n_train,:);
X_test = X(n_train+1:end,:);
y_train = y(1:n_train);
y_test = y(n_train+1:end);
mus = [0.1,1,10,100,1000];
lambdas = [0.1,1,10,100,1000];
test_mses = [];
%X = rand(n,para); % Input matrix. Each row: one sample; Each column: one feature

% definition
beta0 = randn(para,1); % initialize beta
mu = 100; % magnitude of reconstruction noise
lambda = 10; % magnitude of lp penalty
p0 = 0.5; % power index, l_p
eps = 5e-2; % avoid divergent gradient
pnormp = @(x) (sum(abs(x).^p0)); % |x|_p^p
pnorm_grad = @(x) p0./(abs(x).^(1-p0)+eps).*(x>0)+p0./(abs(x).^(1-p0)+eps)*(-1).*(x<=0); % gradient of |x|_p^p
u = @(beta) mu*1/(2*n_train)*norm(y_train-X_train*beta)^2 + lambda * pnormp(beta); % potential
u_grad = @(beta) mu/n_train*(-X_train'*y_train + X_train'*X_train*beta) + lambda*pnorm_grad(beta); % gradient of potential
u_train_loss = @(beta) 1/(2*n_train)*norm(y_train-X_train*beta)^2; % training loss
u_test_loss = @(beta) 1/(2*n_test)*norm(y_test-X_test*beta)^2; % testing loss

q = beta0; % initial parameter (position)
p = 0; % initial momentum
L = 5; % number of steps in each path
qs = []; % collection of betas
step = 1e-3; % step size
b = 2000; % total number of cycles
log_m0 = -1; % \mu_m in QHMC
sigma_m = 0; % \sigma_m in QHMC
losses = []; % collection of losses
train_losses = []; % collection of training losses
test_losses = []; % collection of testing losses

%lambda = 10;
% HMC/QHMC
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
    train_loss = u_train_loss(q);
    test_loss = u_test_loss(q);
    train_losses = [train_losses; train_loss];
    test_losses = [test_losses; test_loss];
end

plot(1:(b0+b), log(test_losses))
hold on


