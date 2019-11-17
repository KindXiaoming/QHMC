%%%%%%%%%% In this script, we implement a toy model %%%%%%%%
%%%%%%%%%% double well distribution x^4-4x^2 %%%%%

u = @(x) x.^4-4*x.^2;
u_grad = @(x) 4*x.^3-8*x;
b = 500; % total number of cycles
step = 0.05;
L = 5;
qs = [];
trial = 200;

% mass is no longer fixed in QHMC
log_m0 = 1;
sigma_m = 3;
escape_time = [];

for j=1:trial
    q = sqrt(2);
    p = 0;
    for i=1:b
        logm = normrnd(log_m0,sigma_m);
        m = 10^logm;
        p = normrnd(0,sqrt(m));
        [qf,pf] = leap_frog(u_grad,m,q,p,step,L);
        if mh(u,m,q,qf,p,pf)==1
            qs = [qs; q];
            q = qf;
        end
        if i<b
            if q<0
                escape_time = [escape_time;i];
                break
            end
        else
            escape_time = [escape_time;b];
        end        
    end
end

nums = [];
for i=1:7
    nums = [nums;length(find(escape_time<50*i))]
end

plot(50*(1:7),nums/trial,'-o')
hold on


