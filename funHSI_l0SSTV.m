function [x,errList] = funHSI_l0SSTV(A,y,sizex,lambda,mu,gamma,alpha,maxiter)
%% Initialization

 mu1=mu(1); 
 
 h=sizex(1);m=sizex(2);n=h*m;s=sizex(3);
 
% difference operators (periodic boundary)
D = @(z) cat(4, z([2:end, 1],:,:) - z, z(:,[2:end, 1],:)-z);%按照第四维度，拼接后两者
Dt = @(z) [-z(1,:,:,1)+z(end,:,:,1); - z(2:end,:,:,1) + z(1:end-1,:,:,1)] ...
    +[-z(:,1,:,2)+z(:,end,:,2), - z(:,2:end,:,2) + z(:,1:end-1,:,2)];%逆向操作
% for fftbased diagonilization
Lap = zeros(h,m);
Lap(1,1) = 4; Lap(1,2) = -1; Lap(2,1) = -1; Lap(end,1) = -1; Lap(1,end) = -1;
Lap = fft2(repmat(Lap, [1,1,s]));%其功能是以A的内容堆叠在（MxN）的矩阵B中，B矩阵的大小由MxN及A矩阵的内容决定
    
temp=A'*y;    
% Aty        = A'*y;
% Aty_mat    = reshape(Aty,h*m,s);
% [U,Sig,V]  = MySVD(Aty_mat);
% rk         = 30;
% temp         = U(:,1:rk)*Sig(1:rk,1:rk)*V(:,1:rk)';
% % x1         = x1(:);
temp1=reshape(temp,h,m,s);
v=D(temp1);
tempr=reshape(temp,n,s);
w = v;
B1=zeros(n*s,1); B2=zeros(n*s,1); b3=zeros(n*s,1);   % Bregman Variables

% Create Total variation matrices
Dh=TVmatrix(h,m,'H');
Dv=TVmatrix(h,m,'V');
Dd=opTV1(s);Dd=Dd';
D1=kron(Dd',Dh); D2=kron(Dd',Dv);

x=zeros(n*s,1);

%% Main iteration
for i=1:maxiter
    lastx=x;
    Q=MySoftTh(D1*x+B1,lambda/mu1); 
    R=MySoftTh(D2*x+B2,lambda/mu1);
%     S=NucTh(x+B3,mu2/mu3);+mu3*(S-B3)
   
    %% update j
     B3=reshape(b3,h,m,s);
    X=reshape(x,h,m,s);
    rhs = mu1*(X+B3) + Dt(v-w)/gamma;
    J = real(ifftn((fftn(rhs))./(Lap/gamma+mu1))); 
     j=reshape(J,n*s,1);
     
      %% update v
    v = ProjL10ball(D(J)+w,alpha);

    temp        = A'*y;
    tempr=reshape(temp,n*s,1);

    bigY=tempr+D1'*(mu1*(Q-B1))+D2'*(mu1*(R-B2))+mu1*(j-b3);  
    

 
    [x(:),~]=lsqr(@afun,bigY(:),1e-6,5,[],[],x(:));
   

    B1=B1+D1*x-Q;
    B2=B2+D2*x-R;
%     B3=B3+x-S;
    b3=b3+x-j;
    w = w + D(J) - v;
 errList(i) = norm(x(:)-lastx(:)) / (norm(lastx(:)));
       if rem(i,10)==0    
           fprintf(' %d iteration done of %d \n',i, maxiter);
       end     
end

function y = afun(z,str)
% +mu3*z
zmat=reshape(z,n*s,1);
temp1=reshape(mu1*(D1'*(D1*zmat)),n*s,1);
temp2=reshape(mu1*(D2'*(D2*zmat)),n*s,1);
temp3=reshape(mu1*(zmat),n*s,1);
tt= (A'*(A*z))+temp1+ temp2 +temp3;
        switch str
            case 'transp'
                y = tt;
            case 'notransp'
                y = tt;
        end
end



end


 
%% This is soft thresholding operation
function X= MySoftTh(B,lambda)

   X=sign(B).*max(0,abs(B)-(lambda/2));
end
%% This is nuclear norm thresholding
function X=NucTh(X,lambda)
if isnan(lambda)
    lambda=0;
end
[u,s,v]= svd(X,0);
s1=MySoftTh(diag(s),lambda);
X=u*diag(s1)*v';
end
%% 
% function newZ = SolveLeastSquare(A,Y,Z)
%     [mn,dim]=size(Z);
%     newZ=zeros(mn,dim);
%     for i=1:dim
%         [newZ(:,i),~]=lsqr(A,Y(:,i),1e-6,5,[],[],Z(:,i));                
%                   
%     end   
% end

%% This is a function to make total variation matrix
function opD=opTV1(m)

%Make two vectors of elements -1 and 1 of lengths (m-1)
B=[ -1*ones(m-1,1),ones(m-1,1)]; 
%Make sparse diagonal matrix D of size m-1 by m
%with -1's on zeroth diagonal and 1's on 1st diagonal;
D=spdiags(B,[0,1],m-1,m);
%add a last row of all zeros in D
D(m,:)=zeros(1,m);
%Make it as operator
opD=D; %It will convert to operator.
end

function opD=TVmatrix(m,n,str)

if str=='H' % This will give matrix for Horizontal Gradient
    D = spdiags([-ones(n,1) ones(n,1)],[0 1],n,n);
    D(n,:) = 0;
    D = kron(D,speye(m));
   
elseif str=='V' %This will give matrix for Verticle Gradient
   D = spdiags([-ones(m,1) ones(m,1)],[0 1],m,m);
   D(m,:) = 0;
   D = kron(speye(n),D);
end
opD=D;
end
