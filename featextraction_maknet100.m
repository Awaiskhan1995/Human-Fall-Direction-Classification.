% load digit dataset
digitDatasetPath = fullfile('E:\Fall_dataset_allframes - Copy');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
 [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
%% design CNN
% net=load('ModifiedVgg16.mat'); %% for vgg16
% net=net.mvgg;

net=load('resudual_model3RBNet.mat'); %% for vgg19
net=net.new;

lgraph=layerGraph(net);



numClasses = numel(categories(imdsTrain.Labels));    
newFCLayer = fullyConnectedLayer(numClasses,'Name','NewFc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc',newFCLayer);
newClassLayer = softmaxLayer('Name','NewSoftmax');
lgraph = replaceLayer(lgraph,'softmax',newClassLayer);

newClassLayer1 = classificationLayer('Name','classification');
lgraph = addLayers(lgraph,newClassLayer1);
lgraph = replaceLayer(lgraph,'classification',newClassLayer1);
lgraph = connectLayers(lgraph,'NewSoftmax','classification');

  %% Augmenter
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    %% Resize training and testing data according to network
%     auimds = augmentedImageDatastore([224 224 3],imdsTrain,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
%     auimdsVali = augmentedImageDatastore([224 224 3],imdsValidation,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
    auimds = augmentedImageDatastore([227 227 3],imdsTrain,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
    auimdsVali = augmentedImageDatastore([227 227 3],imdsValidation,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
    
 options = trainingOptions('sgdm',...
        'ExecutionEnvironment','cpu',...
        'MaxEpochs',1,'MiniBatchSize',128,...
        'Shuffle','every-epoch', ...
        'ValidationData',auimdsVali,...
        'InitialLearnRate',0.0001, ...
        'ValidationFrequency', 5, ...
        'Verbose',false, ...
        'Plots','training-progress');
% set training options


% training the network
% TrainedModifiedNet16 = trainNetwork(auimds, lgraph, options);
BottleneckInvertedResCNNwithselfattention = trainNetwork(auimds, lgraph, options);

% save('TrainedModifiedNet16','TrainedModifiedNet16');
save('TrainedBottleneckInvertedResCNNwithselfattention','BottleneckInvertedResCNNwithselfattention');
featureLayer = 'NewFc';
trainingFeatures_F = activations(BottleneckInvertedResCNNwithselfattention, auimds, featureLayer, ...
   'OutputAs', 'rows');
 trainingLabels = imdsTrain.Labels;
 testingFeatures_F = activations(BottleneckInvertedResCNNwithselfattention, auimdsVali, featureLayer, ...
   'OutputAs', 'rows');
 testingLabels = imdsValidation.Labels;
labels=testingLabels; 
save('labels','labels');
testfeatures = testingFeatures_F;
save('testfeat_resudualmodel','testfeatures');