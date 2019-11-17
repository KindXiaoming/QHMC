%%% plot the results from rec_mnist_hmcq

figure('Renderer', 'painters', 'Position', [0 0 650 250])
subaxis(2, 2, 2, 'sh', 0, 'sv', 0, 'padding', -100, 'margin', 0);

h = 100;
w = 100;
index = [2,3,4,6,7,8];
subplot(2,4,1)
imshow(rec_J)
title("ground truth")
subplot(2,4,5)
imshow(rec_J_noise)
title("noisy")
for i=1:6
if i<4
    subplot(2,4,index(i))
else
    subplot(2,4,index(i))
end
imshow(hmcs{i})
if i<4
    title("HMC params"+num2str(i))
else
    title("QHMC params"+num2str(mod(i-1,3)+1))
end
end