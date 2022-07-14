clc,clear; close all
seed = 2015; 
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

%% read data
load Reno150-150-100;
x_dc = C;
x=trans255(x_dc);
[w,h,s] = size(x);
%% Generate measurements
ratio  = 0.05; 
N      = h*w;
A      = PermuteWHT_partitioned(N,s,ratio);
% A      = PermuteWHT2(N,s,ratio);
b     = A*x(:);

% b=b1  + 0.3*randn(size(b1));
%% tenspec
% clear opts;
lambda=[  19  ]; mu=[ 0.05] ; 
imsize = size(x_dc);
dim=imsize(3);
N = imsize(1)*(imsize(2)); % number of pixels

%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = round(0.3*N); % L0 gradient of output image pavia 40.91
maxiter = 100; % maximum number of iterations  
gamma =1/0.05; % stepsize of ADMM  
eta = 0.98; % controling gamma for nonconvex optimization 0.96 35.263983 dB SSIMΪ 0.9513
epsilon = 0.0002*N; % stopping criterion 

for i=1:1
    fprintf('================== %d/4 ===================\n',i)
    tic;
%     [x_rec] = HSSTV_CS(A, x_dc,b, maxiter,lambda,alpha,nu,mu,gamma);
   [ x_rec,errList] = funHSI_l0SSTV(A,b,[w,h,s] ,lambda,mu,gamma,alpha,maxiter);
    toc;
    xrec_ttv=reshape(x_rec,[w,h,s]);   
    
    [mp(i),sm(i),er(i)] = msqia(xrec_ttv/255,x/255);
end
result(i)=calcDenoiseResult( x/255,x/255,double(xrec_ttv/255),'PAV case2 WTNN WNNM SSTV',false );

figure, imshow(xrec_ttv(:,:,1),[]);
figure, imshow(x(:,:,1),[]);

ergascase1 = ErrRelGlobAdimSyn(x/255,xrec_ttv/255);

A=reshape(x/255,w*h,s);
B=reshape(xrec_ttv/255,w*h,s);
msad = mSAD(A,B);
