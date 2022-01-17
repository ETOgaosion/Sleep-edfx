clear all
close all
clc

sample_data_path = './sample_data/';
raw_data_path = './raw_data/';
% raw_data_path = './raw_eog/';

num_sub = 20; % number of subjects
num_sample = 10; % number of samples taken from each subjects
num_stage = 5; % type of sleep stages

sample = [];
target = [];

for idx_sub = 1:num_sub
   load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
   [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
   [~,labels] = max(labels,[],2);
   
   for c = 1:num_stage
       idx_stage = find(labels==c);
       idx_pure = [];
       for idx = 2:length(idx_stage)-1
           if idx_stage(idx-1)+idx_stage(idx+1) == idx_stage(idx)*2
               idx_pure = [idx_pure;idx_stage(idx)];
           end
       end
       %random sampling
       if c==2 && idx_sub==12 %subject12:pure N1 = 9
           data_stage = [data(idx_pure,:);data(1115,:)];
       elseif c==2 && idx_sub==15 %subject15: pure N1 = 8
           data_stage = [data(idx_pure,:);data([188,189],:)];
       else 
           data_stage = data(idx_pure(randperm(length(idx_pure),num_sample)),:);
       end
       
       sample = [sample;data_stage];
       target = [target;c*ones(num_sample,1)];
   end
end

% draw and compare
for i = 1:size(sample,1)
    subplot(10,5,i);
    plot(sample(i,:));
    title(num2str(target(i)));
end


% save([sample_data_path, 'sample2.mat'], 'sample', 'target');
