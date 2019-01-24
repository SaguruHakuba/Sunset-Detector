net = alexnet;
testImages = imageDatastore('ResizedImg\images\images\test','IncludeSubfolders',true,'LabelSource','foldernames');
trainingImages = imageDatastore('ResizedImg\images\images\train','IncludeSubfolders',true,'LabelSource','foldernames');
validationImages = imageDatastore('ResizedImg\images\images\validate','IncludeSubfolders',true,'LabelSource','foldernames');

%transfer layers to new network
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(trainingImages.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%train network
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);

netTransfer = trainNetwork(trainingImages,layers,options);

%%%%
layer = 'softmax';
% trainingFeatures = activations(net, trainingImages, layer);
[score] = activations(netTransfer, testImages, layer);
% validationFeatures = activations(net, validationImages, layer);

%Classify test images
predictedLabels = classify(netTransfer,testImages);
testLabels = testImages.Labels;
accuracy = mean(predictedLabels == testLabels)

ROC = [];
for threshold = linspace(0,1,101)
    tp = sum((score(:,2)>threshold)&(testLabels=='sunset'));
    fp = sum((score(:,2)>threshold)&(testLabels=='nonsunset'));
    fn = sum((score(:,2)<threshold)&(testLabels=='sunset'));
    tn = sum((score(:,2)<threshold)&(testLabels=='nonsunset'));
    tpr = tp/(tp+fn);
    fpr = fp/(fp+tn);
    ROC = [ROC; threshold,tpr,fpr,tp,tn,fp,fn];    
end

figure(140);
hold on;
threshold = ROC(:,1);
FPR = ROC(:,3);
TPR = ROC(:,2);
plot(FPR,TPR, 'b-', 'LineWidth', 2);
plot(FPR,TPR, 'b.', 'MarkerSize', 6, 'LineWidth', 2);
grid;
title(sprintf('ROC for Alexnet Transfer Learning Using Softmax Layer'), 'fontSize', 18);
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
axis([0 1 0 1]);

dist2 = FPR.^2 + (1-TPR).^2;
best_threshold = threshold(find(dist2 == min(dist2)));

save('alexnet_results.mat','netTransfer','score','ROC','best_threshold');