net = alexnet;
testImages = imageDatastore('ResizedImg\images\images\test','IncludeSubfolders',true,'LabelSource','foldernames');
trainingImages = imageDatastore('ResizedImg\images\images\train','IncludeSubfolders',true,'LabelSource','foldernames');
validationImages = imageDatastore('ResizedImg\images\images\validate','IncludeSubfolders',true,'LabelSource','foldernames');

% [trainingImages,testImages] = splitEachLabel(images,0.7,'randomized');

%Extract Image Features
layer = 'fc8';
trainingFeatures = activations(net, trainingImages, layer);
testFeatures = activations(net, testImages, layer);
validationFeatures = activations(net, validationImages, layer);

trainingLabels = trainingImages.Labels;
testLabels = testImages.Labels;
validationLabels = validationImages.Labels;
% 
% %Fit Image Classfier
% classifer = fitcecoc(trainingFeatures, trainingLabels);
% 
% %Classify Test Images
% predictedLabels = predict(classifer, testFeatures);
% 
% accuracy = mean(predictedLabels == testLabels)
trainingLabels = (grp2idx(trainingLabels))*2-3;
testLabels = (grp2idx(testLabels))*2-3;
validationLabels = (grp2idx(validationLabels))*2-3;


save('CNNfeatures.mat', 'trainingFeatures', 'trainingLabels', 'testFeatures', 'testLabels', 'validationFeatures', 'validationLabels');
