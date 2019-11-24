%%% plot the results from rec_mnist_hmcq

figure('Renderer', 'painters', 'Position', [0 0 650 250])
subaxis(2, 2, 2, 'sh', 0, 'sv', 0, 'padding', -100, 'margin', 0);

h = 100;
w = 100;

subplot(2,5,1)
imshow(rec_J)
title("ground truth")

subplot(2,5,6)
imshow(rec_J_noise)
title("noisy")

subplot(2,5,2)
imshow(hmcs{1})
title("HMC 1")

subplot(2,5,3)
imshow(hmcs{2})
title("HMC 2")

subplot(2,5,4)
imshow(hmcs{3})
title("HMC 3")

subplot(2,5,5)
imshow(hmcs{7})
title("NUTS")

subplot(2,5,7)
imshow(hmcs{4})
title("QHMC 1")

subplot(2,5,8)
imshow(hmcs{5})
title("QHMC 2")

subplot(2,5,9)
imshow(hmcs{6})
title("QHMC 3")

subplot(2,5,10)
imshow(hmcs{8})
title("RHMC")