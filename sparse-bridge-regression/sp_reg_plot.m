%%% section 6.2 , Bayesian bridge regression
%%% grid search for paramters, produce Table 2 results.
rng(1);
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
mus = [0.1,1,10,100,1000];
lambdas = [0.1,1,10,100,1000];
test_mses = [];
%X = rand(n,para); % Input matrix. Each row: one sample; Each column: one feature

for ii=0:24
    ii
    beta0 = randn(para,1);
    col = mod(ii,5);
    row = (ii - mod(ii,5))/5;
    mu = mus(col+1);
    lambda = lambdas(row+1);
    p0 = 0.5;
    eps = 5e-2;
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
    step = 1e-3;
    b = 9000; % total number of cycles
    b0 = 1000; % burn in
    log_m0 = -4;
    sigma_m = 0;
    losses = [];
    train_losses = [];
    test_losses = [];
    step2 = 1e-3;

    for i=1:b0
        q = q - step2*u_grad(q);
        qs = [qs; q'];
        losses = [losses; u(q)];
        train_loss = u_train_loss(q);
        test_loss = u_test_loss(q);
        train_losses = [train_losses; train_loss];
        test_losses = [test_losses; test_loss];
    end

    %lambda = 10;
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

    % plot(1:(b0+b), log(losses))
    % hold on
    % plot(1:(b0+b), log(train_losses))
    % plot(1:(b0+b), log(test_losses))

    test_mse = mean(train_losses(1001:end))
    test_mses = [test_mses;test_mse];
    test_mse_std = std(train_losses(1001:end))

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

    shape = size(qs);
    acc = shape(1)/b
end

test_mses

% acc_rate = (length(qs)-b0)/b
% histogram(qs(b0:end,1))

% subplot(1,2,1)
% histogram(qs(b0:end,1),-0.05:0.005:0.05,'Normalization','pdf')
% title('attribute 1')
% xlabel("value")
% ylabel("probability density")
% hold on
% subplot(1,2,2)
% histogram(qs(b0:end,4),-0.05:0.005:0.05,'Normalization','pdf')
% title('attribute 4')
% xlabel("value")
% ylabel("probability density")
% hold on

% hold on
% for i=1:10
%     subplot(2,6,i)
%     hold on
%     histogram(qs(b0:end,i))
% end
% 
% subplot(2,6,11)
% plot(1:(b+b0), losses)
