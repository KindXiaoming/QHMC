%%% HMC/QHMC of the image denoising model

mu = 100;
lambda1 = 1; % zero means no regularization for [A,B]
lambda2 = 10;
p0 = 0.5;
eps = 0.05;
%e0_th = 0.01;
y = rec_J_noise; 
u1 = @(A,B,e) 0.5*mu*norm(y-A*B-e,'fro')^2/(493*517)+0.5*lambda1*(norm(A,'fro')^2+norm(B,'fro')^2)/((493+517)*50);
u2 = @(A,B,e) sum(sum(lambda2*e.^p0.*(e>0)+lambda2*(-e).^p0.*(e<=0)))/(493*517);
U = @(A,B,e) u1(A,B,e) + u2(A,B,e);
U_grad = @(A,B,e) {(mu*(A*B+e-y)*B'+lambda1*A)/(493*517),(mu*A'*(e+A*B-y)+lambda1*B)/(493*517),(mu*(A*B+e-y)+lambda2*p0./(e.^(1-p0)+eps).*(e>0)-lambda2*p0./((-e).^(1-p0)+eps).*(e<=0))/(493*517)};
step = 1e-4;
e = (rec_J_noise - A*B);
iter = 500;
%hmcs = {};
T = 1e-5;

q = {A,B,e};
thetal = q;
qs = {};
mses = [];
loss1 = [];
loss2 = [];
for i=1:iter
    i
    r0 = {randn(size(A))*sqrt(T),randn(size(B))*sqrt(T),randn(size(e))*sqrt(T)};
    a = cellfun(@(x) sum(x.^2),r0,'un',0);
    P2_prop = sum([a{:}]);
    H_prop = U(thetal{:}) + P2_prop/2;
    %uuu1 = U(thetal{:})
    %kkk1 = P2_prop/2
    logu = log(rand())-H_prop;
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
        %vj = 1;
        if vj==-1
            [thetam,rm,x,y,Cprime,sprime] = BuildTree_mult(thetam,rm,logu,vj,j,step,U,U_grad);
        else
            [x,y,thetap,rp,Cprime,sprime] = BuildTree_mult(thetap,rp,logu,vj,j,step,U,U_grad);
        end
        %dif2 = sum(abs(thetam{1}-theta{1}),'all')
        if sprime==1
            C = {C{:},Cprime{:}}; 
            %C
        end
        temp = cellfun(@minus,thetap, thetam,'un',0);
        a1 = cellfun(@(x,y) sum(x.*y),temp,rm,'un',0);
        inner1 = sum([a1{:}]);
        a2 = cellfun(@(x,y) sum(x.*y),temp,rp,'un',0);
        inner2 = sum([a2{:}]);
        s = sprime*(inner1>0)*(inner2>0);
        j = j+1;
        j
        if j>3
            break
        end
    end
    a = randsample(C,1);
    thetam = a{1}{1};
    r = a{1}{2};
    thetal = thetam;
    %mse = norm(rec_J-q{1}*q{2},'fro')^2;
    %mses = [mses;mse];
    loss1 = [loss1;u1(thetal{:})];
    loss2 = [loss2;u2(thetal{:})];
end

hmc = thetal{1}*thetal{2};
hmcs{end+1} = hmc;
plot(1:iter, loss2-0.01)
title('reg')
hold on