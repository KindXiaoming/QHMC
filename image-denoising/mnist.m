%%%% Read in the original image, turn it to gray-level, and add noise to it

rng(1);
%RGB = imread('./mnist/img_9.jpg');
%[xi, RGB] = imread('./building.png', 'png');
%I = ind2gray(xi,RGB);
RGB = imread('./lena.png', 'png');
I = rgb2gray(RGB);
I = im2single(I);
%rec_J = U(:,1:r0)*S(1:r0,1:r0)*(V(:,1:r0))';
rec_J = I;
rec_J_noise = rec_J;
%imshow(rec_J)
%histogram(J)
shape = size(rec_J);
num_el = prod(shape);
num_bad = 50000;
x = randsample(num_el,num_bad);
rec_J_noise(x) = 1;
e_true = rec_J_noise - rec_J;
%imshow(rec_J)

