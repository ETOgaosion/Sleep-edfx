clear all
close all
clc

sample_data_path = './sample_data/';
% raw_data_path = './raw_data/';
raw_data_path = './raw_eog/';

num_sub = 20; % number of subjects
num_stage = 5; % type of sleep stages

sample = [];
target = [];

select = [2,5];
for idx_sub = 1:num_sub
    load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    [~,labels] = max(labels,[],2);
    flags = zeros(num_epoch,1);
    for epoch_idx = 1:num_epoch-1
        if labels(epoch_idx) ~= labels(epoch_idx+1)
            if ismember(labels(epoch_idx),select)
                if ismember(labels(epoch_idx+1),select)
                    flags(epoch_idx) = 1;
                    flags(epoch_idx+1) = 1;
                end
            end
        end
    end
    sample = [sample;data(flags==1,:)];
    target = [target;labels(flags==1,:)];
end
target(target==2)=1;
target(target==5)=2;

% draw and compare
for i = 1:size(sample,1)
    subplot(10,5,i);
    plot(sample(i,:));
    title(num2str(target(i)));
end

save([sample_data_path, 'sample.mat'], 'sample', 'target');