clear all
close all
clc

addpath('./sleepedfx_scripts/');
addpath('./edf_reader/');

data_path = './CAP-normal/';

raw_data_path = './raw_data/';
if(~exist(raw_data_path, 'dir'))
    mkdir(raw_data_path);
end

% Sampling rate of hypnogram (default is 30s)
epoch_time = 30;
% fs = 512; % sampling frequency

Nsub = 16; % numer of subjects

data_all = cell(Nsub,1);
label_all = cell(Nsub,1);
y_all = cell(Nsub,1); % one-hot encoding

% list only healthy (SC) subjects
listing = dir([data_path, '*.edf']);

for i = 1 : numel(listing)
    disp(listing(i).name)
    % target_dir = [data_path, listing(i).name, '/'];
    
    [~,filename,~] = fileparts(listing(i).name);
    sub_id = i;
    % sub_id = str2double(filename(2:3));
    % sub_id = sub_id + 1; % index 0 to 1
    
    % load edf data to get Fpz-Cz, and EOGhorizontal channels
    % edf_file = [target_dir, listing(i).name, '-PSG.edf'];
    edf_file = [data_path, listing(i).name];
    [header, edf] = edfreadUntilDone(edf_file);
    channel_names = header.label;
    
    for c = 1 : numel(channel_names)
        channel_names{c} = strtrim(channel_names{c});
    end
    % chan_ind_eeg = find(ismember(channel_names, 'C4A1'));
    chan_ind_eeg = find(ismember(channel_names, 'C3A2'));
    if(isempty(chan_ind_eeg))
        disp('Oops, wait. Channel not found!');
        pause;
    end
    
    fs = header.frequency(chan_ind_eeg);
%     if(header.frequency(chan_ind_eeg) ~= fs)
%         disp('Oops, wait! Sampling frequency mismatched!');
%         pause;
%     end
    
    chan_data_eeg = edf(chan_ind_eeg, :);
    clear edf header channel_names
    
    % ensure the signal is calibrated to microvolts
%     while max(chan_data_eeg) <= 10
%         disp('Signal calibrated!');
%         chan_data_eeg = chan_data_eeg * 1000;
%     end
    % zero-mean
    chan_data_eeg = chan_data_eeg - mean(chan_data_eeg);
    
    % load hypnogram
    hyp_file = fullfile([data_path, filename, '.txt']);
    hypnogram = edfx_load_hypnogram_v2(hyp_file);
    statis = tabulate(hypnogram(:));
    disp(statis);
    % process times to determine the in-bed duration
    %[chan_data_eeg_new, chan_data_eog_new, hypnogram_new] = edfx_process_time_2chan(target_dir, chan_data_eeg, chan_data_eog, hypnogram, epoch_time, fs);
%     [chan_data_eeg_new, hypnogram_new] = edfx_process_time(raw_data_path, chan_data_eeg, hypnogram, epoch_time, fs);
%     
%     eeg_epochs = buffer(chan_data_eeg_new, epoch_time*fs);
%     eeg_epochs = eeg_epochs';
% 
%     label = edfx_hypnogram2label(hypnogram_new);
%     % excluding Unknown and non-score
%     ind = (label == 0);
%     disp([num2str(sum(ind)), ' epochs excluded.'])
%     label(ind) = [];
%     eeg_epochs(ind,:,:) = [];
%     %eog_epochs(ind,:,:) = [];
% 
%     %data = zeros([size(eeg_epochs), 2]);
%     %data(:,:,1) = eeg_epochs;
%     %data(:,:,2) = eog_epochs;
%     data = eeg_epochs;
%     
%     % since the d
%     y = zeros(numel(label),1);
%     for k = 1 : numel(label)
%         y(k, label(k)) = 1;
%     end
%     
%     if(isempty(data_all{sub_id}))
%         data_all{sub_id} = data;
%         y_all{sub_id} = y;
%     else
%         data_all{sub_id} = cat(1, data_all{sub_id}, data);
%         y_all{sub_id} = [y_all{sub_id}; y];
%     end
%     % clear X_eeg X_eog label y
%     clear X_eeg label y
   
end

patinfo.ch_orig{1} = "C4A1";
%patinfo.ch_orig{1} = "EOGhorizontal";
patinfo.fs = 100;
%patinfo.chlabels = {"EEG", "EOG"};
patinfo.chlabels = {"EEG"};
patinfo.classes = {"W", "N1", "N2", "N3", "R"};

for s = 1 : Nsub
    labels = logical(y_all{s});
    data = data_all{s};
    save([raw_data_path, 'n', num2str(s,'%02d'), '.mat'], 'data', 'labels', 'patinfo', '-v7.3');
end