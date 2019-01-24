clear all
close all
clc

load('CNNfeatures.mat');

kernelScale=32;
boxConstraint=16;
net = fitcsvm(trainingFeatures, trainingLabels, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, 'Standardize', true);
TPR = [];
FPR = [];
svs =[];

% for threshold = linspace(-2,2,101)
%     [label,score] = predict(net, validationFeatures);
%     tp = sum((score(:,2)>threshold)&(validationLabels==1));
%     fp = sum((score(:,2)>threshold)&(validationLabels==-1));
%     fn = sum((score(:,2)<threshold)&(validationLabels==1));
%     tn = sum((score(:,2)<threshold)&(validationLabels==-1));
%     tpr = tp/(tp+fn);
%     fpr = fp/(fp+tn);
%     % fprintf('True positive rate: %.3f\nFalse Positive Rate: %.3f\n',tpr,fpr);
%     TPR =[TPR, tpr];
%     FPR =[FPR, fpr];
%     svs =[svs, sum(net.IsSupportVector)];
%     ROC = [ROC; tpr,fpr,threshold,tp,tn,fp,fn]; 
% end

ROC = [];
[label,score] = predict(net, testFeatures);
for threshold = linspace(-2,2,201)
    [label,score] = predict(net, testFeatures);
    tp = sum((score(:,2)>threshold)&(testLabels==1));
    fp = sum((score(:,2)>threshold)&(testLabels==-1));
    fn = sum((score(:,2)<threshold)&(testLabels==1));
    tn = sum((score(:,2)<threshold)&(testLabels==-1));
    tpr = tp/(tp+fn);
    fpr = fp/(fp+tn);
    % fprintf('True positive rate: %.3f\nFalse Positive Rate: %.3f\n',tpr,fpr);
    ROC = [ROC; tpr,fpr,threshold,tp,tn,fp,fn];    
end

% out = []; %contains kernelScale, boxConstraint, tp, fp, fn, tn, tpr, fpr
% for i = -5:10
%     kernelScale = power(2,i);
%     for j=-5:10
%         fprintf('i = %d, j = %d',i,j);
%         boxConstraint=power(2,j);        
%         net = fitcsvm(trainingFeatures, trainingLabels, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, 'Standardize', true);
%         [label,score] = predict(net,validationFeatures);
%         tp = sum((label>0)&(validationLabels==1));
%         fp = sum((label>0)&(validationLabels==-1));
%         fn = sum((label<0)&(validationLabels==1));
%         tn = sum((label<0)&(validationLabels==-1));
%         tpr = tp/(tp+fn);
%         fpr = fp/(fp+tn);
%         [num_sv, trash] = size(net.SupportVectors);
%         out = [out; kernelScale boxConstraint tp fp fn tn tpr fpr num_sv];
%     end
% end
% save('cnn_hyperparameter_optimization.mat','out');

%Plot the ROC
figure(139);
hold on;
threshold = ROC(:,3);
FPR = ROC(:,2);
TPR = ROC(:,1);
plot(FPR,TPR, 'b-', 'LineWidth', 2);
plot(FPR,TPR, 'b.', 'MarkerSize', 6, 'LineWidth', 2);
grid;
title(sprintf('Kernel Width=%.1f, Box Constraint=%.1f',kernelScale, boxConstraint), 'fontSize', 18);
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
axis([0 1 0 1]);

dist2 = FPR.^2 + (1-TPR).^2;
best_threshold = threshold(find(dist2 == min(dist2)));

save('something_about_threshold.mat','net','ROC','best_threshold');
