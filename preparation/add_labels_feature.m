clear all
close all
clc

%fea_data_path = './fea_par/';
fea_data_path = './subbands_fea/';
dir_fea = dir(fea_data_path);

fea2_data_path = './subbands_fea2/';
if(~exist(fea2_data_path, 'dir'))
    mkdir(fea2_data_path);
end

for idx_sub = 3:length(dir_fea)  % the real index = idx_sub-2
    load([fea_data_path,dir_fea(idx_sub).name]);
    fea = fea(:,7:end);
    seq_num = size(fea,1);
    [~,labels] = max(labels,[],2);  %turn one-hot codes to indeed label
    % start tag = -1; end tag = -2;
    labels = labels-1;
    labels_tag = [-1; labels; -2];
    %labels_tag = labels_tag';
    epo_prev = zeros(seq_num,1);
    epo_next = zeros(seq_num,1);
    pha_prev = zeros(seq_num,1);
    pha_next = zeros(seq_num,1);
    for seq_idx = 1:seq_num
        epo_prev(seq_idx) = labels_tag(seq_idx);
        epo_next(seq_idx) = labels_tag(seq_idx+2);
    end
    
    fea2 = cat(2,fea,epo_prev,epo_next);
    save([fea2_data_path, 'fea', num2str(idx_sub-2,'%02d'), '.mat'], 'fea2', 'labels');
    
end

