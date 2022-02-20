load('TrainingSamplesDCT_8_new.mat')

BGSamples = size(TrainsampleDCT_BG,1);
BGDimension = size(TrainsampleDCT_BG,2);
FGSamples = size(TrainsampleDCT_FG,1);
FGDimension = size(TrainsampleDCT_FG,2);

xDimension = 255;
modifiedXDimension = xDimension-7;
yDimension = 270;
modifiedYDimension = yDimension-7;
EMLimit = 1000;

% Dimesions to perform classification on, as well as the C value for EM.
C = 8;
dimensionList = [1,2,4,8,16,24,32,40,48,56,64];

% Read in the initial image.
initialImage = imread('cheetah.bmp');
initialImage = im2double(initialImage);

% Read in the mask image.
initialMaskImage = imread('cheetah_mask.bmp');
initialMaskImage = im2double(initialMaskImage);

% Convert entire initialImage into dct form to create a 65,224x64 matrix
% which contains the dct transformation of each 64x64 block.
zigZagInitialImage = zeros(modifiedXDimension*modifiedYDimension,64);
for i = 1:modifiedXDimension
    for j = 1:modifiedYDimension
        currentBlock = initialImage(i:i+7,j:j+7);
        dctMatrix = dct2(currentBlock);
        
        % Convert Matrix into Zig-Zag Pattern
        ind = reshape(1:numel(dctMatrix), size(dctMatrix));     %# indices of elements
        ind = fliplr( spdiags( fliplr(ind) ) );                 %# get the anti-diagonals
        ind(:,1:2:end) = flipud( ind(:,1:2:end) );              %# reverse order of odd columns
        ind(ind==0) = [];                                       %# keep non-zero indices
        zigZagInitialImage((i-1)*(modifiedYDimension)+j,:) = dctMatrix(ind);   %# get elements in zigzag order
    end
end

% ---------- BG Expectation-Maximization ----------
% Initialize pi_c by creating 8 random values that all sum to 1.
piEMBG = randi(1, C);           % Start with 3 random numbers that don't sum to 1.
piEMBG = piEMBG / sum(piEMBG);  % Normalize so the sum is 1.

% Initialize mu_c by choosing C random observations in the sample data.
muEMBG = TrainsampleDCT_BG(randi([1 BGSamples],1,C),:);

% Initialize sigma_c by creating an identity matrix of random values.
sigmaEMBG = zeros(BGDimension,BGDimension,C);
for i =1:C
    sigmaEMBG(:,:,i) = (rand(1,BGDimension)).*eye(BGDimension);
end   

BDRJoint = zeros(BGSamples,C);
for i = 1:EMLimit
    % ---------- E-step ----------
    % Compute hIJ by Gaussian BDRJoint for P_Z|X using mu, sigma and pi.
    for j = 1:C
        BDRJoint(:,j) = mvnpdf(TrainsampleDCT_BG,muEMBG(j,:),sigmaEMBG(:,:,j))*piEMBG(j);    
    end
    hIJ = BDRJoint./sum(BDRJoint,2);
    % Calculate the log-likelihood of the resulting BDRJoint data.
    BDRLikelihood(i) = sum(log(sum(BDRJoint,2)));
    
    % ---------- M-step ----------
    % Updated piEMBG value for n+1.
    piEMBG = sum(hIJ)/BGSamples;
    % Updated muEMBG value for n+1.
    muEMBG = (hIJ'*TrainsampleDCT_BG)./sum(hIJ)';
    % Updated sigmaEMBG value for n+1.
    for j = 1:C
        sigmaEMBG(:,:,j) = diag(diag(((TrainsampleDCT_BG-muEMBG(j,:))'.*hIJ(:,j)'* ... 
            (TrainsampleDCT_BG-muEMBG(j,:))./sum(hIJ(:,j),1))+0.0000001));
    end
    
    % If likelihood hasn't changed by more than .1% between iteration stop.
    if i > 1
        if abs(BDRLikelihood(i) - BDRLikelihood(i-1)) < 0.001
            break; 
        end
    end
end
% ---------- BG Expectation-Maximization ----------

% ---------- FG Expectation-Maximization ----------
% Initialize pi_c by creating 8 random values that all sum to 1.
piEMFG = randi(1, C);           % Start with 3 random numbers that don't sum to 1.
piEMFG = piEMFG / sum(piEMFG);  % Normalize so the sum is 1.

% Initialize mu_c by choosing C random observations in the sample data.
muEMFG = TrainsampleDCT_FG(randi([1 FGSamples],1,C),:);

% Initialize sigma_c by creating an identity matrix of random values.
sigmaEMFG = zeros(FGDimension,FGDimension,C);
for i =1:C
    sigmaEMFG(:,:,i) = (rand(1,FGDimension)).*eye(FGDimension);
end   

BDRJoint = zeros(FGSamples,C);
for i = 1:EMLimit
    % ---------- E-step ----------
    % Compute hIJ by Gaussian BDRJoint for P_Z|X using mu, sigma and pi.
    for j = 1:C
        BDRJoint(:,j) = mvnpdf(TrainsampleDCT_FG,muEMFG(j,:),sigmaEMFG(:,:,j))*piEMFG(j);    
    end
    hIJ = BDRJoint./sum(BDRJoint,2);
    % Calculate the log-likelihood of the resulting BDRJoint data.
    BDRLikelihood(i) = sum(log(sum(BDRJoint,2)));
    
    % ---------- M-step ----------
    % Updated piEMFG value for n+1.
    piEMFG = sum(hIJ)/FGSamples;
    % Updated muEMFG value for n+1.
    muEMFG = (hIJ'*TrainsampleDCT_FG)./sum(hIJ)';
    % Updated sigmaEMFG value for n+1.
    for j = 1:C
        sigmaEMFG(:,:,j) = diag(diag(((TrainsampleDCT_FG-muEMFG(j,:))'.*hIJ(:,j)'* ... 
            (TrainsampleDCT_FG-muEMFG(j,:))./sum(hIJ(:,j),1))+0.0000001));
    end
    
    % If likelihood hasn't changed by more than .1% between iteration stop.
    if i > 1
        if abs(BDRLikelihood(i) - BDRLikelihood(i-1))<0.001
            break; 
        end
    end
end
% ---------- FG Expectation-Maximization ----------

% Classification on learned BG and FG mixtures across all 11 dimensions.
for currentDimension = 1:length(dimensionList)
    dimensionK = dimensionList(currentDimension);
    % Compute BDR for EM which is the sum over BDR for each sample in the data.
    maskMatrix = zeros(modifiedXDimension*modifiedYDimension,1);
    for x = 1:length(zigZagInitialImage)
        probabilityBG = 0;
        probabilityFG = 0;

        % Compute total BDR for background data.
        for y = 1:size(muEMBG,1)
            probabilityBG = probabilityBG + mvnpdf(zigZagInitialImage(x,1:dimensionK), ...
                muEMBG(y,1:dimensionK),sigmaEMBG(1:dimensionK,1:dimensionK,y))*piEMBG(y);
        end
        % Compute total BDR for foreground data.
        for y = 1:size(muEMFG,1)
            probabilityFG = probabilityFG + mvnpdf(zigZagInitialImage(x,1:dimensionK), ...
                muEMFG(y,1:dimensionK),sigmaEMFG(1:dimensionK,1:dimensionK,y))*piEMFG(y);
        end
        % Compare BDR between background and foreground.
        if probabilityBG < probabilityFG
            maskMatrix(x) = 1;
        end
    end

    % Reform maskMatrix into a 255x270 matrix.
    tempMask = zeros(modifiedXDimension,modifiedYDimension);
    for x = 1:modifiedXDimension
        tempMask(x,:) = maskMatrix(((x-1)*(modifiedYDimension)+1):x*(modifiedYDimension))';
    end
    maskMatrix = tempMask;
%     figure
%     imshow(maskMatrix,[])

    % Using the maskMatrix and the initialMaskImage calculate PoE.
    incorrectCount = 0;
    for x = 1:modifiedXDimension
        for y = 1:modifiedYDimension
            if initialMaskImage(x,y) ~= maskMatrix(x,y)
                incorrectCount = incorrectCount + 1;
            end
        end
    end
    errorEM1(currentDimension) = incorrectCount/xDimension/yDimension; 
end

hold on;
plot(dimensionList,errorEM1,'o-','linewidth',1,'markersize',5)
plot(dimensionList,errorEM2,'o-','linewidth',1,'markersize',5)
plot(dimensionList,errorEM3,'o-','linewidth',1,'markersize',5)
plot(dimensionList,errorEM4,'o-','linewidth',1,'markersize',5)
plot(dimensionList,errorEM5,'o-','linewidth',1,'markersize',5)
legend('FG1','FG2','FG3','FG4','FG5')