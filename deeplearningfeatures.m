%---Used GPU automatically if cuda-toolkit and cudNN lib installed with microsoft VS 2015
clc;
clear all;
%net= vgg19;
%net = mobilenetv2;
net = densenet201;
%net = resnet50;
%net = inceptionv3;
%%
%   Load Images
rootFolder = fullfile('D:\PhD Projects\Retinal Disease\Practice code\Awais\code');

% classes = {'Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground','Pond','Port','RailwayStation','Resort','River','School','SparseResidential','Square','Stadium','StorageTanks','Viaduct'};

imds = imageDatastore(fullfile(rootFolder,'OCT_SEOUL_2'), 'LabelSource','foldernames', 'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.jpeg','.png','.bmp','.jfif'});

%%
%   Balancing & Counting Images w.r.t all the classes
tbl = countEachLabel(imds);
%minSetCount = min(tbl{:,2});
%%%%%%%%%%%%%imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)
%%
%   Prepare Training and Test Image Sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

countEachLabel(trainingSet)
%%
%   Pre-process Images For CNN
imageSize = net.Layers(1).InputSize;
% imageSize = net.meta.normalization.imageSize(1:3)

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
 augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

%%
%   Prepare Training and Test Labels
trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;

%%
%   Extract Training Features Using CNN
featureLayer = 'avg_pool';
tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeatures.mat
feat=testFeatures';
feat=double(feat);
label=testLabels;
label=double(label);

%%%%%%%%%%%%%%%   ACO
% Actual ho  = 0.2
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);

% Parameter setting
N        = 20; 
max_Iter = 10; 
tau      = 1; 
eta      = 1; 
alpha    = 1; 
beta     = 1; 
rho      = 0.2; 
phi      = 0.5; 
Nf       = 1900;       % Set number of selected features
% Ant Colony System
[sFeat,Nf,Sf,curve] = jACO(feat,label,N,max_Iter,tau,eta,alpha,beta,rho,phi,Nf,HO);

% Plot convergence curve
plot(1:max_Iter,curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ACS'); grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of ACO
%%%%%%%%%%%%%%% Firefly Algorithem
% Data=testFeatures;
% [r c]=size(testFeatures);
% disp('Data---->')
% disp(Data)
% nOfSelection=c;
% nOfFireFlies=50;
% itrationMax=50;
% fun=@meandata;
% [Selection , SelectionValue]=fireflySelection(Data,nOfSelection,nOfFireFlies,itrationMax,fun);
% MinimizedData=Data(:,Selection);
% 
% 
% testLabels=testLabels';
% 
% classifier2 = fitcecoc(sFeat, testLabels, ...
%     'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% 
% %%%%%%%%%%%%%%%    End of Fire fly
% %   Evaluate Classifier
% predictedLabels = predict(classifier2, sFeat, 'ObservationsIn', 'columns');
% 
% % Tabulate the results using a confusion matrix.
% confMat = confusionmat(testLabels, predictedLabels);
% 
% % Convert confusion matrix into percentage form
% confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
% 
% % Display the mean accuracy
% mean(diag(confMat))


%%
%   Saving specifically feature vector from workspace
% skin2_PH2_P_seg_imagenet_resnet152_fc1000 = trainingFeatures';
% save('skin2_PH2_P_seg_imagenet_resnet152_fc1000','skin2_PH2_P_seg_imagenet_resnet152_fc1000')

%%
%   Preparing feature_Vector(x) for classification
x1=sFeat;
y=testLabels;

x = array2table(x1);
x.type = y;


%%
%   Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%%
%   Evaluate Classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Display the mean accuracy
mean(diag(confMat))
toc

%%
%   Saving specifically feature vector from workspace
% skin2_PH2_P_seg_imagenet_resnet152_fc1000 = trainingFeatures';
% save('skin2_PH2_P_seg_imagenet_resnet152_fc1000','skin2_PH2_P_seg_imagenet_resnet152_fc1000')

%%
%   Preparing feature_Vector(x) for classification
x = array2table(testFeatures');
x.type = testLabels;
classificationLearner

testfeaturesorg=x;
save testfeaturesorg.mat


x = array2table(trainingFeatures');
x.type = trainingLabels;

trainingfeaturesorg=x;
save trainingfeaturesorg.mat

% for i=1:length(x)
% sp=max(x)-min(x);
% sp=round(sp);
% sp1(1,:)=sp;
% i=i+1;
% end
classificationLearner

%% Next Part
train =testFeatures';
  train=double(train);
  [r, c]=size(train);
new_score = Find_Entropy(train,c);
ent_FV1 = real(new_score(:,1:600));

train =ent_FV1;
 x=array2table(train);
x.type=testLabels;

%%
%   Try the Newly Trained Classifier on Test Images
% newImage = imread(fullfile(rootFolder, 'Images', '1WwO9.jpg'));
newImage=imread('D:\WCE\660.jpg');

% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% Make a prediction using the classifier
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

[r c j]=size(newImage);

position = [(r/6),2];
value = char(label(1));
%RGB = insertText(img,position,value);

RGB=insertText(newImage,position,value,'FontSize',28,'TextColor','white','BoxColor','blue','Font','Agency FB Bold');
subplot(121),imshow(newImage),title('Input Image');
subplot(122),imshow(RGB),title('Predicted Class');

 y=cellstr(testLabels);
% y = species;
X=testFeatures';
% Create a random partition for a stratified 10-fold cross-validation.

c = cvpartition(y,'leaveout',1);
% Create a function that computes the number of misclassified test samples.

fun = @(xTrain,yTrain,xTest,yTest)(sum(~strcmp(yTest,...
    classify(xTest,xTrain,yTrain)))); 
% Return the estimated misclassification rate using cross-validation.

rate = sum(crossval(fun,X,y,'partition',c))...
           /sum(c.TestSize)
% y=trainingLabels;
% c2 = cvpartition(y,'KFold',10);
% fun = @(trainingFeatures,trainingLabels,testFeatures,testLabels)(sum(~strcmp(testLabels,...
%     classify(testFeatures,trainingFeatures,trainingLabels))));
% rate = sum(crossval(fun,trainingFeatures,y,'partition',c2))...
%            /sum(c.TestSize);



%%%%%%%%%%%%%%%%%              Code for testing the image belong to which
%%%%%%%%%%%%%%%%%              clasc
%newImage = imread(fullfile('1.jpg'));
%newImage = imread(fullfile('99 (8).jpg'));
 %newImage = imread(fullfile('2.jpg'));
 newImage = imread(fullfile('cscr.jpg'));
 %newImage = imread(fullfile('4-1.jpg'));
%newImage = imread(fullfile('5-1.jpg'));
%newImage = imread(fullfile('5641231 (237).jpg'));
ds = augmentedImageDatastore(imageSize,testSet, 'ColorPreprocessing', 'gray2rgb');
ds = augmentedImageDatastore(imageSize,newImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, ds, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
sprintf('The input image belongs to the %s class', label)

% %%%%%%%%%   gradcam
% label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
% sprintf('The input image belongs to the %s class', label)
% imds = imread("cscr.jpg");
% inputSize = net.Layers(1).InputSize(1:2);
% imds = imresize(imds,inputSize);
% imshow(imds)
% label = classify(net,imds)
% %label = categorical
%     % toy poodle 
%      scoreMap = gradCAM(net,imds,label);
%      figure
% imshow(imds)
% hold on
% imagesc(scoreMap,'AlphaData',0.5)
% colormap jet
% % % %%%%%%%%% end gradcam

%figure
 figure
subplot(2,2,1);
imshow(newImage)
title("Input image ");

subplot(2,2,2);
imshow(newImage);
title(sprintf('The input image belongs to the %s class', label));
%%%%%%%%%%%% This is the end of the testing code