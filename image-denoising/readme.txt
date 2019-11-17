To reproduce Figure 12, here are the following steps:
(1) run mnist.m to read in image (note the directory of the image)
(2) run rec_mnist.m to initialize the model with SVD of the corrupted image
(3) run rec_mnist_hmcq.m to sample images with HMC/QHMC
(4) run rec_mnist_hmcq_plot display all images