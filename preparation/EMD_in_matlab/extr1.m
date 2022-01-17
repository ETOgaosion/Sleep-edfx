%extr
function [indmin,indmax,indzer]=extr1(x)
% [DATAfile DATApath]=uigetfile('*.txt',' ‰»Î–≈∫≈');
% watchon;
% 
% FILENAME=[DATApath,DATAfile];
% x=load(FILENAME);

numzer=1;
nummin=1;
nummax=1;
for g=2:(length(x)-1)

if (x(g-1)*x(g)<0)|(x(g-1)*x(g)==0)&(x(g-1)==0)
indzer(numzer)=g-1;
numzer=numzer+1;
elseif (x(g-1)*x(g)<0)|(x(g-1)*x(g)==0)&(x(g)==0)
indzer(numzer)=g;
numzer=numzer+1;
elseif (x(g-1)*x(g)<0)|(x(g-1)*x(g)==0)&(x(g-1)~=0)&(x(g)~=0)
indzer(numzer)=(2*g-1)/2;
numzer=numzer+1;
end

if (x(g-1)>x(g))&(x(g)>x(g+1))
continue;
elseif (x(g-1)>x(g))&(x(g)<x(g+1))
indmin(nummin)=g;
nummin=nummin+1;
elseif (x(g-1)<x(g)) & (x(g)<x(g+1))
continue;
elseif (x(g-1)<x(g))&(x(g)>x(g+1))
indmax(nummax)=g;
nummax=nummax+1;
end
end
% % subplot(2,3,1)
% % plot(indzer,x(indzer),'r-+')
% % title('indzer')
% % pause
% % subplot(2,3,2)
% % plot(indmin,x(indmin),'b-o')
% % title('indmin')
% % pause
% % subplot(2,3,3)
% % plot(indmax,x(indmax),'g-*')
% % title('indmax')
% pause
% %subplot(2,3,4)
% plot(1:length(x),x,'k-s')
% title('x')
% pause
% 
% envmax = interp1(indmax,x(indmax),1:length(x),'spline');
% envmin = interp1(indmin,x(indmin),1:length(x),'spline');
% %subplot(2,3,5)
% plot(1:length(x),envmin,'y-d')
% title('envmin')
% pause
% %subplot(2,3,6)
% plot(1:length(x),envmax,'c-h')
% title('envmax')
% pause
% plot(1:length(x),mean((envmax+envmin)/2),'m-p')
% title('mean((envmax+envmin)/2)')%extr
