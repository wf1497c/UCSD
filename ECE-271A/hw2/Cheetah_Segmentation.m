close all; clear; clc;

%% Data import
img = imread('cheetah.bmp');
[row,col] = size(img);
imshow(img);
zz_order = dlmread('Zig-Zag Pattern.txt');
load('TrainingSamplesDCT_8_new.mat');
N = 8;

%% Problem 1: Priors
% Priors of background and foreground are directly acquired by calculate
% the ratio of traing samples in each class

c_size = size(TrainsampleDCT_FG,1);
g_size = size(TrainsampleDCT_BG,1);

Prior_FG = c_size / (c_size + g_size);
Prior_BG = g_size / (c_size + g_size);

disp('Prior probability of cheetah:')
disp(Prior_FG)
disp('Prior probability of grass:')
disp(Prior_BG)

%% Problem 2: Conditional Probability Given Classes
% Conditional probabilities of X given cheetah/grass can be acquired by
% the histogram of DCT images, which contains locations of 2nd highest
% energy value within each 8*8 blocks in training samples.
% The conditional probabilities are ratio of each X. 

mu_FG_64 = mean(TrainsampleDCT_FG);
sigma_FG_64 = std(TrainsampleDCT_FG);
mu_BG_64 = mean(TrainsampleDCT_BG);
sigma_BG_64 = std(TrainsampleDCT_BG);

figure
for r = 1:N
    for c = 1:N
        subplot(N,N, (r-1) * N + c)
        Gaussian_Plot(mu_FG_64((r-1) * N + c), sigma_FG_64((r-1) * N + c), ...
            mu_BG_64((r-1) * N + c), sigma_BG_64((r-1) * N + c));
    end
end

%best8_i = [1,18,25,27,30,38,40,42];
best8_i = [1:5,40,30,50];


%% Class-conditional Distributions: 64 dim
cov_BG_64 = cov(TrainsampleDCT_BG);
cov_FG_64 = cov(TrainsampleDCT_FG);
alpha_BG_64 = (2*pi)^64 * det(cov_BG_64);
alpha_FG_64 = (2*pi)^64 * det(cov_FG_64);
inv_cov_BG_64 = pinv(cov_BG_64);
inv_cov_FG_64 = pinv(cov_FG_64);

%% Class-conditional Distributions: 8 dim
BG_8dim = TrainsampleDCT_BG(:,best8_i);
FG_8dim = TrainsampleDCT_FG(:,best8_i);
mu_FG_8 = mean(FG_8dim);
sigma_FG_8 = std(FG_8dim);
mu_BG_8 = mean(BG_8dim);
sigma_BG_8 = std(BG_8dim);
cov_BG_8 = cov(BG_8dim);
cov_FG_8 = cov(FG_8dim);
alpha_BG_8 = (2*pi)^8 * det(cov_BG_8);
alpha_FG_8 = (2*pi)^8 * det(cov_FG_8);
inv_cov_BG_8 = pinv(cov_BG_8);
inv_cov_FG_8 = pinv(cov_FG_8);

%% Problem 3: Mask Generation

img_mask = zeros([row,col]);
% Take left top pixel in a block as origin
% The last N-1 pixels are taken as padding
for r = 1:row-N+1
    for c = 1:col-N+1
        sub_img = img(r:r+N-1, c:c+N-1); % 8*8 block image
        sub_img_DCT = dct2(sub_img); 
        sub_img_vec = DCT_zz(sub_img_DCT, zz_order); % The ouput DCT coefficients vector following zz order
        P_x_given_BG = -0.5 * (sub_img_vec - mu_BG_64) * inv_cov_BG_64 * (sub_img_vec - mu_BG_64)' - 0.5 * log(alpha_BG_64);
        P_x_given_FG = -0.5 * (sub_img_vec - mu_FG_64) * inv_cov_FG_64 * (sub_img_vec - mu_FG_64)' - 0.5 * log(alpha_FG_64);
        P_BG_64 = P_x_given_BG + log(Prior_BG);
        P_FG_64 = P_x_given_FG + log(Prior_FG);
        if(P_FG_64 > P_BG_64)
            img_mask(r,c) = 1;
        end
    end
end

A = uint8(img_mask) * 255;

figure
imagesc(A)
title('Predicted Mask')
colormap(gray(255))

%% 
img_mask = zeros([row,col]);
% Take left top pixel in a block as origin
% The last N-1 pixels are taken as padding
for r = 1:row-N+1
    for c = 1:col-N+1
        sub_img = img(r:r+N-1, c:c+N-1); % 8*8 block image
        sub_img_DCT = dct2(sub_img); 
        sub_img_vec = DCT_zz(sub_img_DCT, zz_order); % The ouput DCT coefficients vector following zz order
        sub_img_vec = sub_img_vec(best8_i);
        P_x_given_BG_8 = -0.5 * (sub_img_vec - mu_BG_8) * inv_cov_BG_8 * (sub_img_vec - mu_BG_8)' - 0.5 * log(alpha_BG_8);
        P_x_given_FG_8 = -0.5 * (sub_img_vec - mu_FG_8) * inv_cov_FG_8 * (sub_img_vec - mu_FG_8)' - 0.5 * log(alpha_FG_8);
        P_BG_8 = P_x_given_BG_8 + log(Prior_BG);
        P_FG_8 = P_x_given_FG_8 + log(Prior_FG);
        if(P_FG_8 > P_BG_8)
            img_mask(r,c) = 1;
        end
    end
end

for i = 1:8
    img_mask(:, col-N+i) = 0;
    img_mask(row-N+i, :) = 0;
end
A = uint8(img_mask) * 255;

figure
imagesc(A)
title('Predicted Mask')
colormap(gray(255))
%% Problem 4: Error Rate
Error_Rate(A)

