clear,clc, 

%a = gpuDevice
%close all
datapath='E:\Fall_dataset_allframes - Copy';
imds=imageDatastore(datapath,  'IncludeSubfolders',true, 'LabelSource','foldernames');
total_split=countEachLabel(imds)
[imdsTrain,imdsTest] = splitEachLabel(imds,.8,'randomized');
numClasses = numel(categories(imdsTrain.Labels))


% net=load('utintdense.mat');

net=densenet201;
%net=vgg16;
%net=resnet101;
%net = efficientnetb0;
% plot(mobnet_Transfer)

net.Layers(1)
net.Layers(end)
% Number of class names for ImageNet classification task
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'avg_pool';
%featureLayer = 'fc7';
%featureLayer = 'fc1000';
%featureLayer = 'pool5';
%featureLayer = 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'; %for effieicnt net
% trainingFeatures_F = activations(net, augmentedTrainingSet, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');
% trainingLabels = imdsTrain.Labels;
% 
% classifier = fitcecoc(trainingFeatures_F, trainingLabels, ...
%     'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns')
testFeatures_E = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns', 'ExecutionEnvironment', 'cpu');

% testFeatures_E = activations(net, augmentedTestSet, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');
% predictedLabels = predict(classifier, testFeatures_E, 'ObservationsIn', 'columns');
testLabels = imdsTest.Labels;
densenet201=testFeatures_E;
labels=testLabels; 


save('densenet201','densenet201');
save('labels','labels');
feat=densenet201';
feat=double(feat);
% label=testLabels;
% label=double(label);


% save('trainingFeatures_E','trainingFeatures_E');
% save('testFeatures_E','testFeatures_E');
% save('trainingLabels','trainingLabels');
% save('testLabels','testLabels');
%  
