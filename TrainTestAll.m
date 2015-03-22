
function [xTra,yTra,xTsa,yTsa] = TrainTestAll(data_org,trainratio,tnm)

for i=1:tnm
[m,n]=size(data_org);   %m examples, the first(n-1) colums are features, and the last colum is labels 
d=n-1;                  %feature dimension

%separate the data into training and test parts randomly
numtr=round(trainratio*m);  
rad=randperm(m);
tr_num=rad(1:numtr);      
ts_num=rad(numtr+1:m);

data_train=data_org(tr_num,:);
data_test=data_org(ts_num,:);

xTra(:,:,i)=data_train(:,1:d)';   %training data with each colum corresponding to one training data point
yTra(:,:,i)=data_train(:,n)';     %a row vector containing the class labels for the training data

xTsa(:,:,i)=data_test(:,1:d)';   %test data
yTsa(:,:,i)=data_test(:,n)';     %test label

end
%save TrainTestData.mat xTra yTra xTsa yTsa
end
