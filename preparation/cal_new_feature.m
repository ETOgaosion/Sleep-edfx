clear all
close all
clc

%fea_data_path = './fea_par/';
fea_data_path = './subbands_fea/';
dir_fea = dir(fea_data_path);

%fea2_data_path = './shapelets_fea/';
fea2_data_path = './mmd_fea/';

fea_new_path = './fea_new/';
if(~exist(fea_new_path, 'dir'))
    mkdir(fea_new_path);
end

for idx_sub = 3:length(dir_fea)  % the real index = idx_sub-2
    load([fea_data_path,dir_fea(idx_sub).name]);
    temp = fea;
    
    load([fea2_data_path,dir_fea(idx_sub).name]);
    fea = cat(2,temp,fea);
    
    %����Ԥ������matlab�Դ���mapminmax��ѵ�����Ͳ��Լ���һ������[0,1]֮��
    [fea_scale,ps]=mapminmax(fea',0,100); 
    fea=fea_scale'; 
    
    save([fea_new_path, 'fea', num2str(idx_sub-2,'%02d'), '.mat'], 'fea', 'labels');
    disp(['Success restore:fea', num2str(idx_sub-2,'%02d'), '.mat']);
end