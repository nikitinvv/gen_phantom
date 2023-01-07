N=512;%size of the image for inversion
Ntheta=3*N/4;%number of angular samples
Ns=N;%number of radial samples; Ns==N for this test;
[f,ellipse]=phantom(N);filter_kind='hamming';%ramp,shepp-logan,cosine,cosine2,hamming,hann
ff=apply_filter_2d_exact(f,filter_kind,ellipse);ff=single(ff');
%filtered Radon data
h=apply_filter_exact(Ntheta,Ns,filter_kind,ellipse);h=single(h);

fid=fopen('../data/f','wb');
fwrite(fid,flipud(ff),'single');
fclose(fid);
fid=fopen('../data/R','wb');
fwrite(fid,flipud(h')*2*N*sqrt(2)/sqrt(Ntheta)*sqrt(N),'single');
fclose(fid);
% exit
