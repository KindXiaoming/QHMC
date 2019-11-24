%%% section 6.2, (log) loss of testing mse
%%% to reproduce Figure 11 (a), one needs to run this script twice
%%% by setting sigma_m=0(HMC) and sigma_m>2(QHMC)

rng(1);

% read data and standarize data
diabetes = importdata('diabetes.txt');
data = diabetes.data;
n = 442;
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
%X = rand(n,para); % Input matrix. Each row: one sample; Each column: one feature

% definition
beta0 = randn(para,1); % initialize beta
mu = 100; % magnitude of reconstruction noise
lambda = 10; % magnitude of lp penalty
p0 = 0.5; % power index, l_p
eps = 0.05; % avoid divergent gradient
pnormp = @(x) (sum(abs(x).^p0)); % |x|_p^p
pnorm_grad = @(x) p0./(abs(x).^(1-p0)+eps).*(x>0)+p0./(abs(x).^(1-p0)+eps)*(-1).*(x<=0); % gradient of |x|_p^p
U = @(beta) mu*1/(2*n_train)*norm(y_train-X_train*beta)^2 + lambda * pnormp(beta); % potential
U_grad = @(beta) mu/n_train*(-X_train'*y_train + X_train'*X_train*beta) + lambda*pnorm_grad(beta); % gradient of potential
u_train_loss = @(beta) 1/(2*n_train)*norm(y_train-X_train*beta)^2; % training loss
u_test_loss = @(beta) 1/(2*n_test)*norm(y_test-X_test*beta)^2; % testing loss

thetal = beta0; % initial parameter (position)
step = 0.05; % step size
b = 2000; % total number of cycles
losses = []; % collection of losses
train_losses = []; % collection of training losses
test_losses = []; % collection of testing losses

for m=1:b
    m
    r0 = randn(para,1);
    u = rand()*exp(-U(thetal)-sum(0.5.*r0.*r0,'all'));
    thetam = thetal;
    thetap = thetal;
    rm = r0;
    rp = r0;
    j = 0;
    C = {};
    C{end+1} = {thetal,r0};
    s = 1;
    while s==1
        vj = randsample([-1,1],1);
        if vj==-1
            [thetam, rm, x, y, Cprime, sprime] = BuildTree(thetam,rm,u,vj,j,step,U,U_grad);
        else
            [x,y,thetap,rp,Cprime,sprime] = BuildTree(thetap,rp,u,vj,j,step,U,U_grad);
        end
        if sprime==1
            C = {C{:},Cprime{:}}; 
        end
        s = sprime*(sum((thetap-thetam).*rm,'all')>0)*(sum((thetap-thetam).*rp,'all')>0);
        j = j+1;
        j
    end
    a = randsample(C,1);
    thetam = a{1}{1};
    r = a{1}{2};
    thetal = thetam;
    %xs{end+1} = thetal;
    loss = U(thetal);
    losses = [losses; loss];
    train_loss = u_train_loss(thetal);
    test_loss = u_test_loss(thetal);
    train_losses = [train_losses; train_loss];
    test_losses = [test_losses; test_loss];
end


plot(1:b, log(test_losses))
hold on


