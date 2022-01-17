clear all
close all
clc

addpath('./hosa/');

imf_data_path = './imf_data/';
bis_par_path = './imf_bis/';
if(~exist(bis_par_path, 'dir'))
    mkdir(bis_par_path);
end

num_sub = 20; % number of subjects

nfft = 128; %%fft length
nlag = 100;
nsamp = 500; % samples per segment
overlap = 0; % percentage overlap of segments
fs = 100; %²ÉÑùÆµÂÊ

for idx_sub = 1:num_sub
    load([imf_data_path, 'imf', num2str(idx_sub,'%02d'), '.mat']);
    [~,labels] = max(labels,[],2);
    labels = labels-1;
    [num_epoch,num_imf,len_imf] = size(imfs); % number of epochs; number of IMF; length of IMF
    
    %bispectrum estimation
    for epoch_idx = 1:num_epoch
        
        bs_amp = zeros(num_imf,256,256);
        bs_pha = zeros(num_imf,256,256);
        % figure();
        for imf_idx = 1:num_imf
            % high order cumulants
            [Bspec,waxis] = BISPECI(imfs(epoch_idx,imf_idx,:), nlag, nsamp, overlap, flag, nfft); 
            waxis = waxis*fs;
%           waxis2 = waxis1(size(waxis1,1)/2+1:size(waxis1,1));
%           bs1 = abs(Bspec1);
            bs_real = abs(real(Bspec));
            bs_imag = abs(imag(Bspec));
        
            bs_amp(imf_idx,:,:) = bs_real(size(bs_real,1)/2+1:size(bs_real,1),size(bs_real,1)/2+1:size(bs_real,1));
            bs_pha(imf_idx,:,:) = bs_imag(size(bs_imag,1)/2+1:size(bs_imag,1),size(bs_imag,1)/2+1:size(bs_imag,1));
            
%             subplot(num_imf,2,imf_idx*2-1);
%             contour(waxis,waxis,bs_real);
%             subplot(num_imf,2,imf_idx*2);
%             contour(waxis,waxis,bs_imag);
        end
        
        label = labels(epoch_idx,:);
        save([bis_par_path, 'bis', num2str(idx_sub,'%02d'),'/', num2str(epoch_idx), '.mat'], 'bs_amp', 'bs_pha', 'label');
    end
    disp(['Success save:imf_bis', num2str(idx_sub,'%02d')]);
    
%     [~,labels] = max(labels,[],2);
%     labels = labels-1;
%     
%     save([bis_par_path, 'imf_bis', num2str(idx_sub,'%02d'), '.mat'], 'bs_amp', 'bs_pha', 'labels');
%     disp(['Success save:imf_bis', num2str(idx_sub,'%02d'), '.mat']);
end