close all; clear; clc;

%% Data import
img = imread('cheetah.bmp');
imshow(img);
zz_order = dlmread('Zig-Zag Pattern.txt');
load('TrainingSamplesDCT_8.mat');
N = 8;

%% Problem 1: Priors
% Priors of background and foreground are directly acquired by calculate
% the ratio of traing samples in each class

c_size = size(TrainsampleDCT_FG,1);
g_size = size(TrainsampleDCT_BG,1);

P_c = c_size / (c_size + g_size);
P_g = g_size / (c_size + g_size);

disp('Prior probability of cheetah:')
disp(P_c)
disp('Prior probability of grass:')
disp(P_g)

% Feature vector: location of the coefficient of 2nd largest magnitude,
BG_X = feature_vector_DCT(TrainsampleDCT_BG);
FG_X = feature_vector_DCT(TrainsampleDCT_FG);

%% Problem 2: Conditional Probability Given Classes
% Conditional probabilities of X given cheetah/grass can be acquired by
% the histogram of DCT images, which contains locations of 2nd highest
% energy value within each 8*8 blocks in training samples.
% The conditional probabilities are ratio of each X. 

bin_ranges=0.5:1:63.5; % 64 bins for plot
P_X_given_c_vec = DCT_histogram(FG_X,8); % Scale to [0,1]
P_X_given_g_vec = DCT_histogram(BG_X,8); % Scale to [0,1]

figure;
subplot(1,2,1)
bar(bin_ranges,P_X_given_c_vec,'histc')
title('Histogram in Foreground Training Samples')
xlabel('X')
ylabel('P(X|cheetah)')

subplot(1,2,2)
bar(bin_ranges,P_X_given_g_vec,'histc')
title('Histogram in Background Training Samples')
xlabel('X')
ylabel('P(X|grass)')

%% Problem 3: Mask Generation
binary = 0;
sub_img_X_map = zeros(size(img));

% Take left top pixel in a block as origin
% The last N-1 pixels are taken as padding
for r = 1:size(img,1)-N+1
    for c = 1:size(img,2)-N+1
        sub_img = img(r:r+N-1, c:c+N-1); % 8*8 block image
        sub_img_DCT = dct2(sub_img); 
        sub_img_X = X_by_DCT(sub_img_DCT, zz_order); % The ouput is a scalar, which indicates X
        sub_img_X_map(r,c) = sub_img_X; % A map recorded each X in all pixels
    end
end

A = zeros(size(img));
for r = 1:size(img,1)-N+1                           
    for c=1:size(img,2)-N+1
        if P_X_given_c_vec(sub_img_X_map(r,c)) * P_c > P_X_given_g_vec(sub_img_X_map(r,c)) * P_g
            A(r,c)=1;
        else
            A(r,c)=0;
        end
    end
end
A = uint8(A) * 255;

figure
imagesc(A)
title('Predicted Mask')
colormap(gray(255))

%% Problem 4: Error Rate
ground_t = imread('cheetah_mask.bmp');
err = 0;
for r = 1:size(A,1)
    for c = 1:size(A,2)
        if ground_t(r,c) ~= A(r,c)
            err = err + 1;
        end
    end
end

disp('Error rate(%):');
disp(err / size(A,1) / size(A,2) * 100);

