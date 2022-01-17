% execute MHS and analyze its subbands
function fea_sub = tosubband(x,fs)
%fs = 100;T = 30;
%N = T*fs;
amp2_sub = zeros(1,6);
energy_sub = zeros(1,6);
power_sub = zeros(1,6);

imf = emd(x,'MAXITERATIONS',20);
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
      if rang_f(idx_f)>=0.5 && rang_f(idx_f)<=2 && mhs(idx_f)>75  %slow wave
          amp2_sub(1) = amp2_sub(1) + abs(mhs(idx_f)-mhs(idx_f+1));
          energy_sub(1) = energy_sub(1) + mhs(idx_f)^2;
      elseif rang_f(idx_f)>=0.5 && rang_f(idx_f)<4  %delta wave
          amp2_sub(2) = amp2_sub(2) + abs(mhs(idx_f)-mhs(idx_f+1));
          energy_sub(2) = energy_sub(2) + mhs(idx_f)^2;
      elseif rang_f(idx_f)>=4 && rang_f(idx_f)<8  %theta wave
          amp2_sub(3) = amp2_sub(3) + abs(mhs(idx_f)-mhs(idx_f+1));
          energy_sub(3) = energy_sub(3) + mhs(idx_f)^2;
      elseif rang_f(idx_f)>=8 && rang_f(idx_f)<13  %alpha wave
          amp2_sub(4) = amp2_sub(4) + abs(mhs(idx_f)-mhs(idx_f+1));
          energy_sub(4) = energy_sub(4) + mhs(idx_f)^2;
      elseif rang_f(idx_f)>=13 && rang_f(idx_f)<30  %beta wave
          amp2_sub(5) = amp2_sub(5) + abs(mhs(idx_f)-mhs(idx_f+1));
          energy_sub(5) = energy_sub(5) + mhs(idx_f)^2;
      elseif rang_f(idx_f)>=30  %gamma wave
          amp2_sub(6) = amp2_sub(6) + abs(mhs(idx_f)-mhs(idx_f+1));
          energy_sub(6) = energy_sub(6) + mhs(idx_f)^2;
      end
  end
  
  %Step3: calculate the spectrum entropy of every subwave
  [pxx, fxx] = pburg(imf(l,:),20,[],fs);
  for idx_f = 1:size(fxx)
      if fxx(idx_f)>=0.5 && fxx(idx_f)<=2 && mhs(idx_f)>75  %slow wave
          power_sub(1) = power_sub(1) + pxx(idx_f) * log(1/pxx(idx_f));
      elseif fxx(idx_f)>=0 && fxx(idx_f)<4  %delta wave
          power_sub(2) = power_sub(2) + pxx(idx_f) * log(1/pxx(idx_f));
      elseif fxx(idx_f)>=4 && fxx(idx_f)<8  %theta wave
          power_sub(3) = power_sub(3) + pxx(idx_f) * log(1/pxx(idx_f));
      elseif fxx(idx_f)>=8 && fxx(idx_f)<13  %alpha wave
          power_sub(4) = power_sub(4) + pxx(idx_f) * log(1/pxx(idx_f));
      elseif fxx(idx_f)>=13 && fxx(idx_f)<30  %beta wave
          power_sub(5) = power_sub(5) + pxx(idx_f) * log(1/pxx(idx_f));
      elseif fxx(idx_f)>=30  %gamma wave
          power_sub(6) = power_sub(6) + pxx(idx_f) * log(1/pxx(idx_f));
      end
  end
  
  fea_sub = cat(2,amp2_sub,energy_sub,power_sub);
%   subplot(k,1,l);
%   plot(rang_f*fs,temp);
%   ylabel(['MHS-',num2str(l)]);
end