clear all
close all
clc

data_path = './pku_database/pku_h5/';
% filename = 'duliyan-2.h5'; file_idx = 1;
% filename = 'guofenglong-2.h5'; file_idx = 2;
% filename = 'niepeipei-2.h5'; file_idx = 3;
% filename = 'wangqing-2.h5'; file_idx = 4;
% filename = 'zhangaijing-2.h5'; file_idx = 5;
% filename = 'zhangxi-2.h5'; file_idx = 6;

% h5disp(filename,'/');
raw_data_path = './pku_database/pku_raw/';
data = h5read([data_path,filename],'/data');
data = data(:, 4, :);  %F4-M1 Channel
% data = data(:, 6, :);  %C4-M1 Channel
% data = data(:, 8, :);  %O2-M1 Channel
data = squeeze(data);
data = data';

%zero-mean
data = data-mean(data);
%ensure the signal is calibrated to microvolts
if(max(data) <= 10)
    disp('Signal calibrated!');
    data = data * 1000;
end
if(max(data) <= 10)
    disp('Signal calibrated!');
    data = data * 1000;
end

labels = h5read([data_path,filename],'/label');
for l = 1:length(labels)
    if labels(l) == 5
        labels(l) = 4;
    end
end
save([raw_data_path, 'pku', num2str(file_idx, '%02d'), '.mat'],'data','labels');
disp(['Success restore raw EEG:','pku', num2str(file_idx, '%02d'), '.mat']);

