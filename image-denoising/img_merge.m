rng(1);
RGB = imread('./mnist/img_9.jpg');
%I = rgb2gray(RGB);
J = im2single(RGB);
[U,S,V] = svd(J);
r0 = 10;
rec_J = U(:,1:r0)*S(1:r0,1:r0)*(V(:,1:r0))';
rec_J_noise = rec_J;
%imshow(rec_J)
%histogram(J)
shape = size(rec_J);
num_el = prod(shape);
num_bad = 100;
x = randsample(num_el,num_bad);
rec_J_noise(x) = 1;
%imshow(rec_J)
RGB = imread('./fig/image/hmc_4.jpg');
hmc = im2single(RGB);
RGB = imread('./fig/image/qhmc_44.jpg');
hmcq = im2single(RGB);

%figure
figure('Renderer', 'painters', 'Position', [10 10 500 500])
subaxis(2, 2, 2, 'sh', 0, 'sv', 0, 'padding', 0, 'margin', 0);
subplot(2,2,1)
imshow(rec_J)
title('Original','Fontsize',15)
subplot(2,2,2)
imshow(rec_J_noise)
title('Noise','Fontsize',15)
subplot(2,2,3)
imshow(abs(e_hmc))
title({'Error (HMC)','mse=0.72'},'Fontsize',15)
subplot(2,2,4)
imshow(abs(e_qhmc))
title({'Error (QHMC)','mse=0.19'},'Fontsize',15)