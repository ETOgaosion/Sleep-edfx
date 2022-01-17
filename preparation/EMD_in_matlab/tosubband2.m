% execute MHS and analyze its subbands
function fea_sub = tosubband2(x,fs)
%fs = 100;T = 30;
%N = T*fs;
amp2_eog = 0;
energy_eog = 0;
power_eog = 0;

imf = emd(x,'MAXITERATIONS',10,'MAXMODES',5);
imf = imf(1:end-1,:);
k = size(imf,1);

%figure;
for l = 1:k
  %Hilbert Transform
  [A,fa,tt] = hhspectrum(imf(l,:));
  NN = length(tt);
  %calculate Marginal Hilbert Spectrum
  fa = fa*fs;
  delt_f = (max(fa)-min(fa))/NN;
  rang_f = min(fa):delt_f:max(fa);
  mhs = zeros(1,length(rang_f));
  for i = 1:length(fa)-1
    for j = 1:length(rang_f)-1
        if (fa(i)>rang_f(j)) && (fa(i)<rang_f(j+1))
            mhs(j) = mhs(j)+A(i);
        else
            mhs(j) = mhs(j);
        end
    end
  end
  
  %Step1: calculate the peak to peak amptitude of every subwave
  %Step2: calculate the subwave energy of MHS
  for idx_f = 1:length(rang_f)-1
      amp2_eog = amp2_eog + abs(mhs(idx_f)-mhs(idx_f+1));
      energy_eog = energy_eog + mhs(idx_f)^2;
  end
  
  %Step3: calculate the spectrum entropy of every subwave
  [pxx, fxx] = pburg(imf(l,:),20,[],fs);
  for idx_f = 1:size(fxx)
      power_eog = power_eog + pxx(idx_f);
  end
  
  fea_sub = cat(2,amp2_eog,energy_eog,power_eog);
%   subplot(k,1,l);
%   plot(rang_f*fs,temp);
%   ylabel(['MHS-',num2str(l)]);
end