clear all
clc

% %iris dataset
data_org=xlsread('Iris.xls');
[m,n]=size(data_org);

d=n-1;                   %input dimension
c=max(data_org(:,n));    %number of classes
r=c-1;

%%%%%================================%%%%%%%%%%%%
%% Data preparation         
tnm=1;                    %running times
trainratio=0.7;           %training and test spliting ratio

%% separate the original data into training and test part
[xTra,yTra,xTsa,yTsa] = TrainTestAll(data_org,trainratio,tnm);

sigma=0.001;              %tolerance used in bisection search
deltal=0;                 %lower bound of /delta
deltau=20;                %upper bound of  /delta  

%% calculate the transform matrix W by SD-WLDA algorithm
tic
for pp=1:tnm
xTr=xTra(:,:,pp);
yTr=yTra(:,:,pp);
xTs=xTsa(:,:,pp);
yTs=yTsa(:,:,pp);

s1=toc;
sdtW(:,:,pp) = SD_WLDA(xTr,yTr,c,r,sigma,deltal,deltau);
runtime(pp,1)=toc-s1;
end
