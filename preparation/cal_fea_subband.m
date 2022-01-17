clear all
close all
clc

addpath('./EMD_in_matlab/');
wave_label = {'slow','delta','theta','alpha','beta','gamma'};
%wave_label = {'delta','theta','alpha','beta','gamma'};
raw_data_path = './raw_data/';
fea_par_path = './subbands_fea/';
if(~exist(fea_par_path, 'dir'))
    mkdir(fea_par_path);
end

num_sub = 20; % number of subjects

for idx_sub = 1:num_sub
    load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    fea = zeros(num_epoch, numel(wave_label)*3);
    for idx_epoch = 1:num_epoch
        fea(idx_epoch,:) = tosubband2(data(idx_epoch,:),patinfo.fs);
    end
    [~,labels] = max(labels,[],2);
    labels = labels-1;
    save([fea_par_path, 'fea', num2str(idx_sub,'%02d'), '.mat'], 'fea', 'labels');
    disp(['Success restore:fea', num2str(idx_sub,'%02d'), '.mat']);
end