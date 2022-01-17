clear all
close all
clc

addpath('./hosa/');

raw_data_path = './raw_data/';
fea_par_path = './mmd_fea/';
if(~exist(fea_par_path, 'dir'))
    mkdir(fea_par_path);
end

num_sub = 20; % number of subjects

maxlag = 50; % maximum lag to be computed
nsamp = 1500; % samples per segment
overlap = 0; % percentage overlap of segments
flag='biased';

len_win = 100; % length of sliding window

for idx_sub = 1:num_sub
    load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    
    cross_zeros = zeros(num_epoch,1);
%     sig_3_cums = zeros(num_epoch,1);
%     sig_4_cums = zeros(num_epoch,1);
    %cross zero points
    for epoch_idx = 1:num_epoch
        for temp_t = 1:len_epoch-1
            if data(epoch_idx,temp_t)*data(epoch_idx,temp_t+1) <= 0
                cross_zeros(epoch_idx) = cross_zeros(epoch_idx)+1;
            end
        end
        
        % high order cumulants
        for k = -maxlag:maxlag
            sig3cum(:,k+maxlag+1) = cum3est(data(epoch_idx,:), maxlag, nsamp, overlap, flag, k);
            % sig_4_cums(epoch_idx) = CUM4EST(data(epoch_idx,:), maxlag, nsamp, overlap, flag, k);
        end
        
        %显示三阶累积量
        figure();
        %等高线图
        subplot(2,2,1);
        contour(-maxlag:maxlag,-maxlag:maxlag,sig3cum);
        xlabel('延迟量k');
        ylabel('延迟量l');
        %三维图
        subplot(2,2,2);
        mesh(-maxlag:maxlag,-maxlag:maxlag,sig3cum);
        xlabel('延迟量k');
        ylabel('延迟量l');
    end
    
    % maximum-minimum distance
    num_win = len_epoch/len_win;
    data = reshape(data,num_epoch,len_win,num_win);
    data = permute(data,[1,3,2]);
    
    [M,p] = max(data,[],3);
    [m,q] = min(data,[],3);
    mmd = sqrt((M-m).*(M-m)+(p-q).*(p-q));
    
    mmd = sum(mmd,2);
    
    fea = cat(2,mmd,cross_zeros);
    
    [~,labels] = max(labels,[],2);
    labels = labels-1;
    
    % save([fea_par_path, 'fea', num2str(idx_sub,'%02d'), '.mat'], 'fea', 'labels');
    % disp(['Success restore:fea', num2str(idx_sub,'%02d'), '.mat']);
end
