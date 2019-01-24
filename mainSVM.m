% Example of using a datastore, see 

clc;

rootdir = '/Users/boutell/ManualBackup/Sunset/Crawled/images/';
subdir = [rootdir 'train'];

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Make datastores for the validation and testing sets similarly.

fprintf('Read images into datastores\n');

xTrain = imageDatastoreReader(trainImages);
yTrain = trainImages.Labels;

%% Train and evaluate an SVM







