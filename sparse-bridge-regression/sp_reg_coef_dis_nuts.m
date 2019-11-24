%%% section 6.2 , Bayesian bridge regression
%%% grid search for paramters, produce Table 2 results.
rng(1);
t = cputime;
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

X_train = X(1:n_train,:);
X_test = X(n_train+1:end,:);
y_train = y(1:n_train);
y_test = y(n_train+1:end);
mu = 100;
lambda = 100;
%X = rand(n,para); % Input matrix. Each row: one sample; Each column: one feature

beta0 = randn(para,1);
p0 = 0.5;
eps = 0.001;
pnormp = @(x) (sum(abs(x).^p0));
pnorm_grad = @(x) p0./(abs(x).^(1-p0)+eps).*(x>0)+p0./(abs(x).^(1-p0)+eps)*(-1).*(x<=0);
u = @(beta) mu*1/(2*n_train)*norm(y_train-X_train*beta)^2 + lambda * pnormp(beta);
u_grad = @(beta) mu/n_train*(-X_train'*y_train + X_train'*X_train*beta) + lambda*pnorm_grad(beta);
u_train_loss = @(beta) 1/(2*n_train)*norm(y_train-X_train*beta)^2;
u_test_loss = @(beta) 1/(2*n_test)*norm(y_test-X_test*beta)^2;

q = beta0;
p = 0;
L = 5;
qs = [];
step = 0.1;
b = 1000; % total number of cycles
b0 = 1000; % burn in
losses = [];
train_losses = [];
test_losses = [];
step2 = 1e-4;

for i=1:b0
    q = q - step2*u_grad(q);
    %qs = [qs; q'];
    u(q)
    losses = [losses; u(q)];
    train_loss = u_train_loss(q);
    test_loss = u_test_loss(q);
    train_losses = [train_losses; train_loss];
    test_losses = [test_losses; test_loss];
end

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
            [thetam, rm, x, y, Cprime, sprime] = BuildTree(thetam,rm,u,vj,j,eps,U,U_grad);
        else
            [x,y,thetap,rp,Cprime,sprime] = BuildTree(thetap,rp,u,vj,j,eps,U,U_grad);
        end
        if sprime==1
            C = {C{:},Cprime{:}}; 
        end
        s = sprime*(sum((thetap-thetam).*rm,'all')>0)*(sum((thetap-thetam).*rp,'all')>0);
        j = j+1;
    end
    a = randsample(C,1);
    thetam = a{1}{1};
    r = a{1}{2};
    thetal = thetam;
    qs = [qs; thetal'];
end


%x = categorical({'AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6'});
x = 1:10;
y = mean(qs);
y_err = std(qs);

errhigh = y + y_err;
errlow  = y - y_err;
%bar(x,y)                
hold on
er = errorbar(x+0.3,y,errlow,errhigh,'d');    
alpha(0.4)

xticks(1:10)
xticklabels({'AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6'})
plot(0:11,zeros(1,12),'r--')
for j=0:10
    plot(ones(1,3)*(j+0.5),-1:1,'k')
end
time = cputime-t