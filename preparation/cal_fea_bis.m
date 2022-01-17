clear all
close all
clc

addpath('./hosa/');

raw_data_path = './raw_data/';
fea_par_path = './bis_fea/';
if(~exist(fea_par_path, 'dir'))
    mkdir(fea_par_path);
end

num_sub = 20; % number of subjects

nfft = 128; %%fft length
nlag = 100;
nsamp = 500; % samples per segment
overlap = 0; % percentage overlap of segments
fs = 100; %²ÉÑùÆµÂÊ

for idx_sub = 1:num_sub
    load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
    % data = data([8,13,20,32,195],:);
    [num_epoch,len_epoch] = size(data); % number of epochs; length of a epoch
    
    bs_amp = zeros(num_epoch,256,256);
    bs_pha = zeros(num_epoch,256,256);
    %bispectrum estimation
    for epoch_idx = 1:num_epoch
        % high order cumulants
        [Bspec,waxis] = BISPECI(data(epoch_idx,:), nlag, nsamp, overlap, flag, nfft); 
        waxis = waxis*fs;
%         waxis2 = waxis1(size(waxis1,1)/2+1:size(waxis1,1));
%         bs1 = abs(Bspec1);
        bs_real = abs(real(Bspec));
        bs_imag = abs(imag(Bspec));
        
        bs_amp(epoch_idx,:,:) = bs_real(size(bs_real,1)/2+1:size(bs_real,1),size(bs_real,1)/2+1:size(bs_real,1));
        bs_pha(epoch_idx,:,:) = bs_imag(size(bs_imag,1)/2+1:size(bs_imag,1),size(bs_imag,1)/2+1:size(bs_imag,1));
%         bs2 = bs1(size(bs1,1)/2+1:size(bs1,1),size(bs1,1)/2+1:size(bs1,1));
%         figure();
%         subplot(1,2,1);
%         mesh(waxis1,waxis1,bs_real);
%         subplot(1,2,2);
%         mesh(waxis1,waxis1,bs_imag);
        
    end
    
    [~,labels] = max(labels,[],2);
    labels = labels-1;
    
    save([fea_par_path, 'fea', num2str(idx_sub,'%02d'), '.mat'], 'bs_amp', 'bs_pha', 'labels');
    disp(['Success save:fea', num2str(idx_sub,'%02d'), '.mat']);
end