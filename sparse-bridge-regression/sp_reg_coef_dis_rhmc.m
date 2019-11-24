%%% section 6.2, (log) loss of testing mse
%%% to reproduce Figure 11 (a), one needs to run this script twice
%%% by setting sigma_m=0(HMC) and sigma_m>2(QHMC)

rng(1);
t = cputime;
% read data and standarize data
diabetes = importdata('diabetes.txt');
data = diabetes.data;
n = 442;
n_train = 300; % number of samples
n_test = 142;
para = 10; % number of parameters
X = data(:,1:10);
reg = 0;
invM = transpose(X)*X+reg*eye(para);
M = inv(invM);
M = (M+transpose(M))/2;
%sqrtM = sqrtm(M);
%sqrtM = (sqrtM+transpose(sqrtM))/2;
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
%X = rand(n,para); % Input matrix. Each row: one sample; Each column: one feature

% definition
beta0 = randn(para,1); % initialize beta
mu = 100; % magnitude of reconstruction noise
lambda = 10; % magnitude of lp penalty
p0 = 0.5; % power index, l_p
eps = 0.001; % avoid divergent gradient
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
step = 2e-5; % step size
b = 2000; % total number of cycles
losses = []; % collection of losses
train_losses = []; % collection of training losses
test_losses = []; % collection of testing losses

for i=1:b
    i
    p = transpose(mvnrnd(zeros(para,1),M));
    [qf,pf] = rhmc_leap_frog(u_grad,invM,q,p,step,L);
    if rhmc_mh(u,invM,q,qf,p,pf)==1
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

%x = categorical({'AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6'});
x = 1:10;
y = mean(qs);
y_err = std(qs);

errhigh = y + y_err;
errlow  = y - y_err;
%bar(x,y)                
hold on
er = errorbar(x+0.1,y,errlow,errhigh,'d');    
alpha(0.4)

xticks(1:10)
xticklabels({'AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6'})
%plot(0:11,zeros(1,12),'r--')
time = cputime-t

