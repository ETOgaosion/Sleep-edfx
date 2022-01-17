clear all
close all
clc

addpath('./EMD_in_matlab/');

num_sub = 20; % number of subjects
num_imf = 4; % number of selected top mhss
fs = 100; NN=2998;
wave_label = {'delta','theta','alpha','beta','gamma'};

imf_data_path = './imf_data/';
dir_imf = dir(imf_data_path);
mhs_data_path = './mhs_data/';
dir_mhs = dir(mhs_data_path);

fea_par_path = './fea_par/';
if(~exist(fea_par_path, 'dir'))
    mkdir(fea_par_path);
end

for idx_sub = 3:length(dir_imf)  % the real index = idx_sub-2
    load([imf_data_path,dir_imf(idx_sub).name]);
    load([mhs_data_path,dir_mhs(idx_sub).name]);
    
    %Step1&2: calculate the max frequency and the corresponding amp energy of IMF
    %Step3: calculate the spectrum entropy of IMF
    %Step4&5&6: calculate the skewness,kurtosis and AUC of MHS
    %Step7: calculate the subwave energy of MHS
    [num_epoch,num_imf,~] = size(imfs);
    %num_mhs = num_imf;
    
    est_f = zeros(num_epoch, 1);
    max_f = zeros(num_epoch, num_imf);  %store the max frequency of IMF
    max_amp = zeros(num_epoch, num_imf);  %store the corresponding amp energy of IMF
    se_imf = zeros(num_epoch, num_imf);  %store the specturm entropy of IMF
    skew_mhs = zeros(num_epoch, num_imf);  %store the skewness of MHS
    kurt_mhs = zeros(num_epoch, num_imf);  %store the kertosis of MHS
    auc_mhs = zeros(num_epoch, num_imf);  %store the area under curve of MHS
    Esub_mhs = zeros(num_epoch, numel(wave_label));  %store the sub-waves energy MHS
    
    for idx_epoch = 1:num_epoch
        IMF = squeeze(imfs(idx_epoch,:,:));
        [A,fa,tt]=hhspectrum(IMF);
        [Nx,X,n,nv,mx,f] = stats_imf(fa);
        est_f(idx_epoch) = f;
        max_f(idx_epoch,:) = mx;  %Step1 end
        max_amp(idx_epoch,:) = (A(n)).^2;  %Step2 end
        
        for idx_imf = 1:num_imf
           temp = 0;
           pxx = pburg(IMF(idx_imf,:),20,[],fs);
           for p = 1:size(pxx)
               temp = temp + pxx(p) * log(1/pxx(p));
           end
           se_imf(idx_epoch,idx_imf) = temp;  %Step3 end
           
           skew_mhs(idx_epoch,idx_imf) = skewness(mhss(idx_epoch,idx_imf,:));  %Step4 end
           kurt_mhs(idx_epoch,idx_imf) = kurtosis(mhss(idx_epoch,idx_imf,:));  %Step5 end
               
           delta_f = (Mfs(idx_epoch,idx_imf)-mfs(idx_epoch,idx_imf))/NN;
           rang_f = mfs(idx_epoch,idx_imf):delta_f:Mfs(idx_epoch,idx_imf);
           auc_mhs(idx_epoch,idx_imf) = trapz(rang_f,mhss(idx_epoch,idx_imf,:));  %Step6 end
               
           for idx_f = 1:length(rang_f)
               if rang_f(idx_f)>=0.5 && rang_f(idx_f)<4  %delta wave
                   Esub_mhs(idx_epoch,1) = Esub_mhs(idx_epoch,1) + mhss(idx_epoch,idx_imf,idx_f)^2;
               elseif rang_f(idx_f)>=4 && rang_f(idx_f)<8  %theta wave
                   Esub_mhs(idx_epoch,2) = Esub_mhs(idx_epoch,2) + mhss(idx_epoch,idx_imf,idx_f)^2;
               elseif rang_f(idx_f)>=8 && rang_f(idx_f)<14  %alpha wave
                   Esub_mhs(idx_epoch,3) = Esub_mhs(idx_epoch,3) + mhss(idx_epoch,idx_imf,idx_f)^2;
               elseif rang_f(idx_f)>=14 && rang_f(idx_f)<30  %beta wave
                   Esub_mhs(idx_epoch,4) = Esub_mhs(idx_epoch,4) + mhss(idx_epoch,idx_imf,idx_f)^2;
               elseif rang_f(idx_f)>=30 && rang_f(idx_f)<45  %gamma wave
                   Esub_mhs(idx_epoch,5) = Esub_mhs(idx_epoch,5) + mhss(idx_epoch,idx_imf,idx_f)^2;    
               end  %Step7 end
           end
        end
    end
    
    %Concatenate and store the feature matrix
    fea = cat(2,max_f,max_amp,se_imf,skew_mhs,kurt_mhs,auc_mhs,Esub_mhs);
    save([fea_par_path, 'fea', num2str(idx_sub-2,'%02d'), '.mat'], 'fea', 'labels');
end