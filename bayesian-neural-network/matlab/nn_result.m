size = length(qs);
if size>100
    qs = qs(end-100:end);
    size = 100;
end

count_1 = []; % >0.05
count_2 = []; % >0.1
mses_1 = [];
mses_2 = [];
for i=1:size
    count_1 = [count_1; sum(sum(abs(qs{i})<0.05))];
    count_2 = [count_2; sum(sum(abs(qs{i})<0.1))];
    mse1 = 0;
    mse2 = 0;
    W1 = qs{i};
    W1(abs(qs{i})<0.05) = 0;
    W2 = qs{i};
    W2(abs(qs{i})<0.1) = 0;
    for j=1:N
        yp = X(:,j) + max(0,W1*X(:,j));
        mse1 = mse1 + u_err(Yt(:,j),yp,X(:,j),W1);
        yp = X(:,j) + max(0,W2*X(:,j));
        mse2 = mse2 + u_err(Yt(:,j),yp,X(:,j),W2);
    end
    mses_1 = [mses_1;mse1];
    mses_2 = [mses_2;mse2];
end

histogram(count_1)
hold on
histogram(count_2)
mean(count_1)
mean(count_2)
mean(mses_1)/2000
mean(mses_2)/2000

    
