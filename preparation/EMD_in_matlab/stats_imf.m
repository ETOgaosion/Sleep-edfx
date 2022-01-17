function [Nx,X,n,nv,mx,f] = stats_imf(fa)
[l_f,~]=size(fa);
fs = 100;
 for k=1:l_f
     [N0,X0]=hist(fa(k,:)*fs,100);
     Nx(k,:)=N0;X(k,:)=X0;
 end
 for k=1:l_f
     [nv(k),n(k)]=max(Nx(k,:));
     mx(k)=X(k,n(k));
 end
 f=sum(mx)/l_f;