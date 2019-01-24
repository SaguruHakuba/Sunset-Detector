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

kernelScale=32;
boxConstraint=128;
net = fitcsvm(xTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, 'Standardize', true);

% out = []; %contains kernelScale, boxConstraint, tp, fp, fn, tn, tpr, fpr
% for i = -5:10
%     kernelScale = power(2,i);
%     for j=-5:10
%         boxConstraint=power(2,j);        
%         net = fitcsvm(xTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, 'Standardize', true);
%         [label,score] = predict(net,xValidate);
%         tp = sum((label>0)&(yValidate==1));
%         fp = sum((label>0)&(yValidate==-1));
%         fn = sum((label<0)&(yValidate==1));
%         tn = sum((label<0)&(yValidate==-1));
%         tpr = tp/(tp+fn);
%         fpr = fp/(fp+tn);
%         [num_sv, trash] = size(net.SupportVectors);
%         out = [out; kernelScale boxConstraint tp fp fn tn tpr fpr num_sv];
%     end
% end

ROC = [];
for threshold = linspace(-2,2,201)
    [label,score] = predict(net, xTest);
    tp = sum((score(:,2)>threshold)&(yTest==1));
    fp = sum((score(:,2)>threshold)&(yTest==-1));
    fn = sum((score(:,2)<threshold)&(yTest==1));
    tn = sum((score(:,2)<threshold)&(yTest==-1));
    tpr = tp/(tp+fn);
    fpr = fp/(fp+tn);
    % fprintf('True positive rate: %.3f\nFalse Positive Rate: %.3f\n',tpr,fpr);
    ROC = [ROC; tpr,fpr,threshold,tp,tn,fp,fn];    
end

%Plot the ROC
figure(139);
hold on;
threshold = ROC(:,3);
FPR = ROC(:,2);
TPR = ROC(:,1);
plot(FPR,TPR, 'b-', 'LineWidth', 2);
plot(FPR,TPR, 'bo', 'MarkerSize', 6, 'LineWidth', 2);
grid;
title(sprintf('Kernel Width=%.1f, Box Constraint=%.1f',kernelScale, boxConstraint), 'fontSize', 18);
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
axis([0 1 0 1]);

%Determine which threshold produces the shortest distance from (0,1) to
%(fpr,tpr)
dist2 = FPR.^2 + (1-TPR).^2;
best_threshold = threshold(find(dist2 == min(dist2)));

save('final_net_and_threshold.mat','net','best_threshold');

