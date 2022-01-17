clear all
close all
clc

addpath('./EMD_in_matlab/');

num_sub = 6; % number of subjects

fs = 100;
wave_label = {'slow','delta','theta','alpha','beta','gamma'};

raw_data_path = './pku_database/pku_raw/';
dir_mhs = dir(raw_data_path);

fea_par_path = './pku_fea/';
if(~exist(fea_par_path, 'dir'))
    mkdir(fea_par_path);
end

for idx_sub = 3:length(dir_mhs)  % the real index = idx_sub-2
    load([raw_data_path,dir_mhs(idx_sub).name]);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    fea = zeros(num_epoch, numel(wave_label)*3);
    for idx_epoch = 1:num_epoch
        fea(idx_epoch,:) = tosubband(data(idx_epoch,:),fs);
    end
    save([fea_par_path, 'fea', num2str(idx_sub-2,'%02d'), '.mat'], 'fea', 'labels');
    disp(['Success restore:fea', num2str(idx_sub-2,'%02d'), '.mat']);
end