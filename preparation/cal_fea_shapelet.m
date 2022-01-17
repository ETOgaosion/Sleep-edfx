clear all
close all
clc

addpath('./FLAG_functions/');

% raw_data_path = './raw_data/';
raw_data_path = './raw_eog/';
sample_data_path = './sample_data/';
fea_par_path = './shapelets_fea/';
if(~exist(fea_par_path, 'dir'))
    mkdir(fea_par_path);
end

load([sample_data_path, 'shapelets.mat']);

num_sub = 20; % number of subjects

for idx_sub = 1:num_sub
    load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    
    % z-normalize all the timeseries
%     for k = 1:num_epoch
%         data(k,:) = zscore(data(k,:));
%     end
    
    fea = transnew(data',shapelets,index);
    
    [~,labels] = max(labels,[],2);
    labels = labels-1;
    
    save([fea_par_path, 'fea', num2str(idx_sub,'%02d'), '.mat'], 'fea', 'labels');
    disp(['Success restore:fea', num2str(idx_sub,'%02d'), '.mat']);
end
