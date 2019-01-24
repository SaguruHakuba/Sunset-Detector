net = netTransfer;

testImages = imageDatastore('Barcelona','IncludeSubfolders',true,'LabelSource','foldernames');
testImages.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

layer = 'softmax';
[score] = activations(netTransfer, testImages, layer);