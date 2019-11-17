%%%% In this script, we apply QHMC to single-layer FC neural network %%%%
rng(1)
n = 40; % dimension of the input
N = 200; % number of samples
X = randn(n,N); % inputs
Wt = randn(n,n)*0.2; % define the teacher network
nzele = 20;
nz_id = datasample(1:n*n, nzele,'Replace',false);
Wt(nz_id) = 1;
Yt = zeros(n,N);  % define the teacher output
Yt = X + max(0,Wt*X);

mu = 10;
lambda = 0;
eps0 = 0.01;
bs = 10;
p0 = 0.5;
%%%%%%%%% Pretrain the model %%%%%%%%%
W = rand(n,n)*0;
u_err = @(yt,y,x,W) mu/(2*n)*sum((yt-y).^2);
u_reg = @(W) lambda/n*sum(sum(W.^p0));
u_grad_err = @(yt,y,x,W) mu/(2*n)*(1+sign(W*x)).*(y-yt)*x';
u_grad_reg = @(W) lambda/n*p0*1./(abs(W).^(1-p0)+eps0);
epoch = 1000;
step = 0.3;
errs = [];
for i = 1:epoch
    id = randsample(1:N,bs);
    grad_err = 0;
    err = 0;
    for j = 1:bs
        x = X(:,id(j));
        y = max(0,W*x) + x;
        yt = Yt(:,id(j));
        grad_err = grad_err + u_grad_err(yt,y,x,W);
        err = err + u_err(yt,y,x,W);
    end
    grad_err = grad_err/bs;
    err = err/bs;
    grad_reg = u_grad_reg(W);
    grad = grad_err + grad_reg;
    grad = min(grad,2);
    W = W - step * grad;
    errs = [errs;err];
end

plot(1:epoch, errs)
set(gca, 'YScale', 'log')

%%%%%%%% Model Parameters %%%%%%%%%
mu = 10; % the magnitude for the L2 error part
lambda = 10; % the regularizer for the weight matrix
eps0 = 1e-2; % avoid divergence when calculating gradients
therm_m = 1;
%%%%%%%%% Initialization %%%%%%%%%%%%
u_err = @(yt,y,x,W) mu/(2*n)*sum((yt-y).^2);
u_reg = @(W) lambda/n*sum(sum(abs(W.^p0)));
u_grad_err = @(yt,y,x,W) mu/(2*n)*(1+sign(W*x)).*(y-yt)*x';
u_grad_reg = @(W) lambda/n*p0*1./(abs(W).^(1-p0)+eps0);
qs = {};
q = zeros(n,1);
p = zeros(n,1);
zeta = zeros(n,1);

%%%%%%%%%% QHMC Parameters %%%%%%%%%%%%
log_m0 = 3;
sigma_m = 0;
epoch = 1100;
L = 5;
step = 0.03;

for i=1:epoch
    logm = normrnd(log_m0,sigma_m);
    m = 10^logm;
    p = normrnd(0,sqrt(m));
    id = randsample(1:N,bs);
    grad_err = 0;
    err = 0;
    for j = 1:bs
        x = X(:,id(j));
        y = max(0,W*x) + x;
        yt = Yt(:,id(j));
        grad_err = grad_err + u_grad_err(yt,y,x,W);
        err = err + u_err(yt,y,x,W);
    end
    grad_err = grad_err/bs;
    err = err/bs;
    grad_reg = u_grad_reg(W);
    grad = grad_err + grad_reg;
    [Wf,pf,zetaf] = leap_frog_nn(n,therm_m,grad,m,W,p,zeta,step,L);
    err = 0; reg = 0; err_f = 0; reg_f = 0;
    for j = 1:bs
        x = X(:,id(j));
        y = max(0,W*x) + x;
        yt = Yt(:,id(j));
        err = err + u_err(yt,y,x,Wf);
        err_f = err + u_err(yt,y,x,Wf);
    end
    err = err/bs; reg = reg/bs;
    reg = u_reg(W);
    reg_f = u_reg(Wf);
    p = exp((reg+err)-(reg_f+err_f));
    if rand()<p
        qs{end+1} = W;
        W = Wf;
    end
end

acc = length(qs)/epoch


    
