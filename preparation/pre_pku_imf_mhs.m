clear all
close all
clc

addpath('./EMD_in_matlab/');

raw_data_path = './pku_database/pku_raw/';
imf_data_path = './imf_data/pku_imf/';
if(~exist(imf_data_path, 'dir'))
    mkdir(imf_data_path);
end
mhs_data_path = './mhs_data/pku_mhs/';
if(~exist(mhs_data_path, 'dir'))
    mkdir(mhs_data_path);
end

num_sub = 6; % number of subjects
num_imf = 4; % number of selected top mhss

fs = 100;
for idx_sub = 1:num_sub
    load([raw_data_path, 'pku', num2str(idx_sub,'%02d'), '.mat']);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    %get IMFand MHS for every epoch
    %len_imf = len_epoch; len_mhs = len_epoch-1;
    imfs = zeros(num_epoch,num_imf,len_epoch);
    mhss = zeros(num_epoch,num_imf,len_epoch-1);
    Mfs = zeros(num_epoch,num_imf); %store the Max instant frequency
    mfs = zeros(num_epoch,num_imf); %store the min instant frequency
    
    for idx_epoch = 1:num_epoch
        [imfs(idx_epoch,:,:),mhss(idx_epoch,:,:),Mfs(idx_epoch,:),mfs(idx_epoch,:)] = tomhs(data(idx_epoch,:),num_imf,len_epoch,fs);
    end
    save([imf_data_path, 'imf',num2str(idx_sub,'%02d'),'.mat'], 'imfs','labels');
    save([mhs_data_path, 'mhs',num2str(idx_sub,'%02d'),'.mat'], 'mhss','Mfs','mfs','labels');
end