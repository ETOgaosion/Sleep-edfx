clear all
close all
clc

addpath('./hosa/');

raw_data_path = './raw_data/';
raw_bis_path = './bis_fea/';
imf_data_path = './imf_data/';

num_sub = 20; % number of subjects

nfft = 128; %%fft length
nlag = 100;
nsamp = 500; % samples per segment
overlap = 0; % percentage overlap of segments
fs = 100; %²ÉÑùÆµÂÊ

%for idx_sub = 1:num_sub
for idx_sub = 5
    load([raw_data_path, 'n', num2str(idx_sub,'%02d'), '.mat']);
    load([imf_data_path, 'imf', num2str(idx_sub,'%02d'), '.mat']);
    [num_epoch,len_epoch] = size(data);
    [~,num_imf,len_imf] = size(imfs);
    % for epoch_idx = 1:num_epoch
    for epoch_idx = 200
        % draw raw EEG
%         figure();
%         plot(data(epoch_idx,:));
%         
%         % draw bispectrum
%         [Bspec,waxis] = BISPECI(data(epoch_idx,:), nlag, nsamp, overlap, flag, nfft); 
%         waxis = waxis*fs;
%         bs_real = abs(real(Bspec));
%         bs_imag = abs(imag(Bspec));
%         figure();
%         subplot(2,2,1);
%         contour(waxis,waxis,bs_real);
%         subplot(2,2,2);
%         mesh(waxis,waxis,bs_real);
%         subplot(2,2,3);
%         contour(waxis,waxis,bs_imag);
%         subplot(2,2,4);
%         mesh(waxis,waxis,bs_imag);
%         
%         % draw first 4 IMF
%         figure();
%         for imf_idx = 1:num_imf
%             subplot(num_imf,1,imf_idx);
%             temp = reshape(imfs(epoch_idx,imf_idx,:),1,len_imf);
%             plot(temp);
%         end
        
        % draw IMF's bispectrum
        figure();
        for imf_idx = 1:num_imf
            temp = reshape(imfs(epoch_idx,imf_idx,:),1,len_imf);
            [Bspec,waxis] = BISPECI(temp, nlag, nsamp, overlap, flag, nfft); 
            waxis = waxis*fs;
            bs_real = abs(real(Bspec));
            bs_imag = atan(imag(Bspec));
            subplot(2,num_imf,imf_idx);
            contour(waxis,waxis,bs_real);
            subplot(2,num_imf,imf_idx+num_imf);
            contour(waxis,waxis,bs_imag);
        end
    end
end