clear all
close all
clc

load('features.mat');

xTrain = X(1:1600,:);
% xTrain = features(1:1600,:);
yTrain = Y(1:1600,:);

xValidate = X(2601:3200,:);
% xValidate = features(2601:3200,:);
yValidate = Y(2601:3200,:);

xTest = X(1601:2600,:);
% xTest = features(1601:2600,:);
yTest = Y(1601:2600,:);

kernelScale=13;
boxConstraint=50;
net = fitcsvm(xTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, 'Standardize', true);
TPR = [];
FPR = [];
svs =[];
for threshold = linspace(-2,2,101)
    [label,score] = predict(net, xValidate);
    tp = sum((score(:,2)>threshold)&(yValidate==1));
    fp = sum((score(:,2)>threshold)&(yValidate==-1));
    fn = sum((score(:,2)<threshold)&(yValidate==1));
    tn = sum((score(:,2)<threshold)&(yValidate==-1));
    tpr = tp/(tp+fn);
    fpr = fp/(fp+tn);
    % fprintf('True positive rate: %.3f\nFalse Positive Rate: %.3f\n',tpr,fpr);
    TPR =[TPR, tpr];
    FPR =[FPR, fpr];
    svs =[svs, sum(net.IsSupportVector)];
end


figure(139);
hold on;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
plot(FPR, TPR, 'bo', 'MarkerSize', 6, 'LineWidth', 2);
grid;
title(sprintf('Kernal=%.1f, BoxConstrain=%.1f',kernelScale, boxConstraint), 'fontSize', 18);
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
axis([0 1 0 1]);
