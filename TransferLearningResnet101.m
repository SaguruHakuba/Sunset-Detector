%<a>href="https://www.mathworks.com/help/nnet/examples/transfer-learning-using-googlenet.html"</a>

net = resnet101;
testImages = imageDatastore('ResizedImg\images\images\test','IncludeSubfolders',true,'LabelSource','foldernames');
trainingImages = imageDatastore('ResizedImg\images\images\train','IncludeSubfolders',true,'LabelSource','foldernames');
validationImages = imageDatastore('ResizedImg\images\images\validate','IncludeSubfolders',true,'LabelSource','foldernames');

testImages.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
trainingImages.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
validationImages.ReadFcn = @(loc)imresize(imread(loc),[224,224]);

%Extract the layer graph from the trained network and plot the layer graph
lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)

%transfer layers to new network
%remove the last three
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

numClasses = numel(categories(trainingImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

%connect new ones back
lgraph = connectLayers(lgraph,'pool5','fc');

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%train network
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);

netTransfer = trainNetwork(trainingImages,lgraph,options);

%Classify validation images
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

save('resnet_results.mat','netTransfer','score','ROC','best_threshold');
