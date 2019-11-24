%%% HMC/QHMC of the image denoising model

mu = 100;
lambda1 = 1; % zero means no regularization for [A,B]
lambda2 = 10;
p0 = 0.5;
eps = 0.1;
%e0_th = 0.01;
invM = S(1:20,1:20);
invM_mean = mean(diag(invM));
invM = invM/invM_mean;
sqrtM = sqrtm(inv(invM));
y = rec_J_noise; 
u1 = @(A,B,e) 0.5*mu*norm(y-A*B-e,'fro')^2/(493*517)+0.5*lambda1*(norm(A,'fro')^2+norm(B,'fro')^2)/((493+517)*50);
u2 = @(A,B,e) sum(sum(lambda2*e.^p0.*(e>0)+lambda2*(-e).^p0.*(e<=0)))/(493*517);
u = @(A,B,e) u1(A,B,e) + u2(A,B,e);
u_grad = @(A,B,e) {(mu*(A*B+e-y)*B'*invM+lambda1*A*invM)/(493*517),(mu*invM*A'*(e+A*B-y)+lambda1*invM*B)/(493*517),(mu*(A*B+e-y)+lambda2*p0./(e.^(1-p0)+eps).*(e>0)-lambda2*p0./((-e).^(1-p0)+eps).*(e<=0))/(493*517)};
L = 6;
step = 0.0001;
e = (rec_J_noise - A*B);
%e(:) = 1;
%e = rand(shape(1),shape(2));
%mag = 0.05;
%A = U(:,1:r)*sqrt(S(1:r,1:r))+mag*randn(shape(1),r);
%B = sqrt(S(1:r,1:r))*transpose(V(:,1:r))+mag*randn(r,shape(2));
iter = 500;
% mass is no longer fixed in QHMC
%log_m0 = -10;
%sigma_m = 1;
%hmcs = {};
T = 1;

%p = {randn(size(A))*sqrt(T),randn(size(B))*sqrt(T),randn(size(e))*sqrt(T)};
q = {A,B,e};
qs = {};
mses = [];
loss1 = [];
loss2 = [];
log_m0 = -8;
sigma_m = 0;
for i=1:iter
logm = normrnd(log_m0,sigma_m);
m = 10^logm;
mag = 0;
p = {normrnd(0,sqrt(m),size(A))*mag,normrnd(0,sqrt(m),size(B))*mag,normrnd(0,sqrt(m),size(e))*mag};
[qf,pf] = leap_frog_mult(u_grad,m,q,p,step,L);
[accpt,flag] = mh_mult(u,m,q,qf,p,pf);
if flag==1
mse = norm(rec_J-q{1}*q{2},'fro')^2;
loss1 = [loss1;u1(q{:})];
loss2 = [loss2;u2(q{:})];
mses = [mses;mse];
continue
end
if accpt==1
qs{end+1} = qf;
q = qf;
end
mse = norm(rec_J-q{1}*q{2},'fro')^2;
mses = [mses;mse];
loss1 = [loss1;u1(q{:})];
loss2 = [loss2;u2(q{:})];
end


hmc = q{1}*q{2};
hmcs{end+1} = hmc;

plot(1:iter, loss2)
title('reg')
legend('HMC 1','HMC 2','HMC 3','QHMC 1','QHMC 2','QHMC 3','NUTS','RHMC')
hold on