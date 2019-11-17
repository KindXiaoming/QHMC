%%% HMC/QHMC of the image denoising model

mu = 100;
lambda1 = 1; % zero means no regularization for [A,B]
lambda2 = 10;
p0 = 0.5;
eps = 0.1;
%e0_th = 0.01;
y = rec_J_noise; 
u1 = @(A,B,e) 0.5*mu*norm(y-A*B-e,'fro')^2/(493*517)+0.5*lambda1*(norm(A,'fro')^2+norm(B,'fro')^2)/((493+517)*50);
u2 = @(A,B,e) sum(sum(lambda2*e.^p0.*(e>0)+lambda2*(-e).^p0.*(e<=0)))/(493*517);
u = @(A,B,e) u1(A,B,e) + u2(A,B,e);
u_grad = @(A,B,e) {(mu*(A*B+e-y)*B'+lambda1*A)/(493*517),(mu*A'*(e+A*B-y)+lambda1*B)/(493*517),(mu*(A*B+e-y)+lambda2*p0./(e.^(1-p0)+eps).*(e>0)-lambda2*p0./((-e).^(1-p0)+eps).*(e<=0))/(493*517)};
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
log_mean = [-8,-9,-10];
sigmas = [0,1];
hmcs = {};
scores = {}
T = 1;

for sigma = 1:2
    for log_mean0 = 1:3
        p = {randn(size(A))*sqrt(T),randn(size(B))*sqrt(T),randn(size(e))*sqrt(T)};
        q = {A,B,e};
        qs = {};
        mses = [];
        loss1 = [];
        loss2 = [];
        log_m0 = log_mean(log_mean0);
        sigma_m = sigmas(sigma);
        for i=1:iter
        i
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
        if i==401
        score = psnr(rec_J, q{1}*q{2});
        end
        if i>401
        score = score + psnr(rec_J, q{1}*q{2});
        end
        end

        %figure
        %subplot(1,2,1)
        %image_hmc = A*B;
        %subplot(1,2,2)
        %hold on
        %histogram(q{3},-0.2:0.005:0.2)
        %title("m\sim 10^{N(4,4^2)}")
        %hold on

        hmc = q{1}*q{2};
        hmcs{end+1} = hmc;
        scores{end+1} = score/100;
        % subplot(2,3,1)
        % imshow(rec_J)
        % hold on
        % subplot(2,3,2)
        % imshow(hmc)
        % hold on
        % subplot(2,3,3)
        % plot(1:iter, mses/(493*517))
        % title('test mse')
        % hold on
        % subplot(2,3,4)
        % plot(1:iter, loss1)
        % title('nuclear')
        % hold on
        % subplot(2,3,5)
        plot(1:iter, loss2)
        title('reg')
        hold on
        %subplot(2,3,6)
        %subplot(1,2,1)
        %histogram(q{3},-1:0.01:1)
        %title('sparse')
        %hold on
        % subplot(1,2,2)
        % scatter(e(:),q{3}(:))
        % histogram(e(:)-q{3}(:),-0.005:0.0002:0.005)
        %hold on
        %shape = size(qs);
        %acc = shape(2)/iter
    end
end