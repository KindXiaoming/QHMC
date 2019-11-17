%%% investigate the dependence of \mu_m and \sigma_m on
%%% acceptance rate and distribution accuracy with lp examples
%%% (not presented in paper)

log_m0 = 5;
x_mse = [];
x_acc = [];
origin = [];
for sigma_m=[0,2,4,6,8,10,15,20]
    sigma_m
    accs = [];
    mses = [];
    for run=1:20
        [mse,acc] = lp_test(log_m0,sigma_m)
        x_mse = [x_mse;mse];
        x_acc = [x_acc;acc];
        origin = [origin;sigma_m];
    end
end
        