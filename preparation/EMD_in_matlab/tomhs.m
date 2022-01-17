% execute EMD and draw the MHS
function [imf,mhs,Mf,mf] = tomhs(x,k,N,fs)
%fs = 100;T = 30;
%N = T*fs;
imf = emd(x,'MAXITERATIONS',20,'MAXMODES',k);
imf = imf(1:k,:);

mhs = zeros(k,N-1);
Mf = zeros(k,1); mf = zeros(k,1);
%figure;
for l = 1:k
  %Hilbert Transform
  [A,fa,tt] = hhspectrum(imf(l,:)); 
  NN = length(tt);
  
  %calculate Marginal Hilbert Spectrum
  delt_f = (max(fa)-min(fa))/NN;
  rang_f = min(fa):delt_f:max(fa);
  temp = zeros(1,length(rang_f)); 
  for i = 1:length(fa)-1
    for j = 1:length(rang_f)-1
        if (fa(i)>rang_f(j)) && (fa(i)<rang_f(j+1))
            temp(j) = temp(j)+A(i);
        else
            temp(j) = temp(j);
        end
    end
  end
  mhs(l,:) = temp;
  Mf(l,1) = max(fa)*fs;
  mf(l,1) = min(fa)*fs;
%   subplot(k,1,l);
%   plot(rang_f*fs,temp);
%   ylabel(['MHS-',num2str(l)]);
end