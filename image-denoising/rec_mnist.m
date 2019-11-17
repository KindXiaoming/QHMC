%%% initialize the model with SVD of the corrupted image

%imshow(rec_J)
r = 20;
mag = 0.0;
[U,S,V] = svd(rec_J_noise);
A = U(:,1:r)*sqrt(S(1:r,1:r))+mag*randn(shape(1),r);
B = sqrt(S(1:r,1:r))*transpose(V(:,1:r))+mag*randn(r,shape(2));
%e = rec_J_noise - A*B;
%e = zeros(shape(1),shape(2));
%histogram(e)

% iter = 10001;
% step = 3e-3;
% mu = 1;
% lambda1 = 0.02;
% lambda2 = 0.05;
% losses = [];
% 
% for i=1:iter
%     if i>2000
%         step = 3e-6;
%     end
%     loss = norm(rec_J_noise-A*B-e, 'fro');
%     A = A - step*mu*(A*B+e-rec_J_noise)*transpose(B) - step*lambda1*A;
%     B = B - step*mu*transpose(A)*(A*B+e-rec_J_noise) - step*lambda1*B;
%     e = e - step*mu*(A*B+e-rec_J_noise);
%     e = e - step*lambda2*sign(e);
%     losses = [losses; loss];
% end
% 
% figure
% subplot(1,2,1)
% plot(1:iter, losses');
% subplot(1,2,2)
% imshow(A*B)
