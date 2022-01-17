sample_data_path = './sample_data/';

sample_split = reshape(sample,6,200,15);
sample_split = permute(sample_split,[1,3,2]);
sample_split = reshape(sample_split,90,200);
target_split = [ones(45,1);2*ones(45,1)];

sample = sample_split;
target = target_split;

save([sample_data_path, 'sample_split.mat'], 'sample', 'target');