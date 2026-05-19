


%% Example 3: Whale Optimization Algorithm (WOA) 
%clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N = 10;     % number of solutions
opts.T = 100;    % maximum number of iterations
% Parameter of WOA
opts.b = 1;
% Load dataset
load densenet201.mat; 
% Divide data into training and validation sets
HO = cvpartition(labels,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('mfo',feat,labels,opts);
% Define index of selected features
sf_idx = FS.sf;
Sfeat=FS.ff
Bestdensenet201featfinal=Sfeat;
save('Bestdensenet201feat','Bestdensenet201featfinal');
Bestdensenet201featfinal=double(Bestdensenet201featfinal);
Bestdensenet201featfinal=array2table(Bestdensenet201featfinal);
Bestdensenet201featfinal.type=labels; 
% Accuracy  
Acc    = jknn(feat(:,sf_idx),labels,opts); 
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations'); 
ylabel('Fitness Value'); 
title('WOA');
