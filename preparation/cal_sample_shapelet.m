clear all
close all
clc

addpath('./FLAG_functions/');

sample_data_path = './sample_data/';

alpha1 = 0.5;
alpha2 = 0.5;

load([sample_data_path, 'sample.mat']);
[num_epoch,len_epoch] = size(sample); % number of epochs; length of a epoch
    
% z-normalize all the timeseries
for k = 1:num_epoch
    sample(k,:) = zscore(sample(k,:));
end
    
% change the one-hot labels to numeric labels 
% [~,labels] = max(labels,[],2);
if min(target)==0
    target = target+1;
end

% shapelets learning
shapelets =[];
index = [];
randn('seed',1);
t = tic;
    
K = length(unique(target));
C = comC(sample, target, K);

for classiter = 1:K
    % v = admmmul(C, classiter, alpha1, alpha2);
    v = ones(size(sample,2),1);
    block = extracts(v);
    [shapeletstmp,indextmp] = AutoShapeletGeneration(block,classiter,sample,target);
    for i = 1:size(block,1)
        tempmat = shapeletstmp{i};
        if size(tempmat,2)>=15
            for j = 1:size(tempmat,1)
                shapelets = [shapelets;mat2cell(tempmat(j,:),[1],[size(tempmat,2)])];
            end
            index = [index;indextmp(i)*ones(size(tempmat,1),1)];
        end
    end
end
    
distime = toc(t);

save([sample_data_path, 'shapelets.mat'], 'shapelets','index');

%draw shapelets
for s = 1:size(index,1)
    subplot(5,5,s);
    plot(shapelets{s});
end

%use z-normalized euclidean distance to transform the data
% D_tr = transnew(sample',shapelets,index);
% figure;
% mesh(D_tr);
% temp = mean(D_tr,2);
% [~,I] = sort(target);
% temp = temp(I);
% plot(temp);
% colorbar;

% D_train = transnew(x_train',shapelets,index);
% D_test = transnew(x_test',shapelets,index);
% 
% %数据预处理，用matlab自带的mapminmax将训练集和测试集归一化处理[0,1]之间
% % [mtrain,ntrain]=size(D_train); 
% % [mtest,ntest]=size(D_test); 
% % dataset=[D_train;D_test]; 
% % [dataset_scale,ps]=mapminmax(dataset',0,1); 
% % dataset_scale=dataset_scale'; 
% % D_train=dataset_scale(1:mtrain,:); 
% % D_test=dataset_scale((mtrain+1):(mtrain+mtest),:);
% 
% %寻找最优c和g
% %c 的变化范围是 2^(-2),2^(-1.5),...,2^(4), g 的变化范围是 2^(-4),2^(-3.5),...,2^(4)
% [bestacc,bestc,bestg] = SVMcgForClass(y_train,D_train,-2,4,-4,4,3,0.5,0.5,0.9);
% 
% %train phase
% cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
% model = svmtrain(y_train,D_train,cmd);
% disp(cmd);
% 
% %test phase
% [y_pred, accuracy, dec_values]=svmpredict(y_test,D_test,model);
% 
% %show classfication result
% cm = confusionmat(y_test,y_pred);