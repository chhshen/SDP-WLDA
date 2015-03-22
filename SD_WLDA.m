
function transformW = SD_WLDA(xTr,yTr,c,r,sigma,deltal,deltau)

%Function of calculating the transformation matrix W for Worst-case Linear Discriminant
%Analysis (WLDA) based on efficient Semidefinite Programming (SDP).
%
% Full description can be found in: 
%   "Worst-Case Linear Discriminant Analysis as Scalable Semidefinite Feasibility Problems." 
%    Hui Li, Chunhua Shen, Anton van den Hengel and Qinfeng Shi. TIP 2015
%
% USAGE
%  transformW = SD_WLDA_sp2(xTr,yTr,c,r,sigma,deltal,deltau)
%
% INPUTS
%   xTr      - a input data matrix with each column corresponding to one training data point
%   yTr      - a column vector containing the class labels for the training data
%   c        - the number of classes in training data
%   r        - the dimensionality we hope to reduced to
%   sigma    - the tolerance in bisection search
%   deltal   - the lower bound of delta in bisection search
%   deltau   - the upper bound of delte in bisection search

% OUTPUTS
%   transformW   - the learned transformation matrix


d=size(xTr,1);    % The dimensionality of training data in original space

%Construct matrix H_st
H_nor=cell(1,d*(d+1)/2);
normH=zeros(d*(d+1)/2,1);
jj=1;
for s=1:d
    for t=1:d
        if (t>s)
        H=sparse([s,s+d,t,t+d],[t,t+d,s,s+d],1,2*d,2*d);
        normH(jj,1)=norm(H,'fro');
        H_nor{1,jj}= H / normH(jj,1);
        jj=jj+1;
        elseif (t==s)
        H=sparse([s,s+d],[t,t+d],1,2*d,2*d);
        normH(jj,1)=norm(H,'fro');
        H_nor{1,jj}= H / normH(jj,1);
        jj=jj+1;
        end  
    end
end

hh=length(H_nor);     
H_aa=cell2mat(H_nor);
H_bb=reshape(H_aa,2*d*2*d,hh);

%Construct matrix Id_bar
Id_bar=eye(d);
normId=norm(Id_bar,'fro');

%Calculate the within-class scatter measure
for i = 1:c
  inx_i = find( yTr==i);
  X_i = xTr(:,inx_i);
  averm(i,:) = mean(X_i,2);
  temp=X_i'-repmat(averm(i,:),length(inx_i),1);
  Sw(:,:,i) = temp'*temp;
end

%Calculate the between-class scatter measure 
bet=0;
for i=1:c
    for j=i+1:c
      bet=bet+1;
      Sb(:,:,bet)=(averm(i,:)-averm(j,:))'*(averm(i,:)-averm(j,:));
    end
end

a = ones(d,d);
b = tril(a);
idx = find(b~=0);
rr = eye(d);
qq  = rr(idx); 

%parametres for L-BFGS-B algorithm
sij=(c*c-c)/2;
usize=c*sij;
m_dual=usize+1+d*(d+1)/2;
l_bbox=zeros(m_dual,1);
l_bbox(1:usize,1)=0;
l_bbox(usize+1:m_dual,1)=-inf;
u_bbox(1:m_dual,1)=+inf;

S_b=reshape(Sb,d*d,sij);  
S_w=reshape(Sw,d*d,c);

%Solve problem (9)for optimal delta and transformation matrix by bisection search 
ite=0;
while (abs(deltau-deltal)/deltal>sigma)
    ite=ite+1;   
    delta(ite) = (deltal+deltau)/2;

    %Construct S_ijk_bar
    tt=0;
    for bet=1:sij
        for k=1:c
         tt=tt+1;
         S_bar=Sb(:,:,bet)-delta(ite)*Sw(:,:,k);
         normS(tt,1)=norm(S_bar','fro');
         S_bar_nor(:,:,tt)= S_bar/ normS(tt,1);
        end
    end

    data=struct('S_b', S_b, 'S_w', S_w, 'normS',normS, 'H_bb',H_bb,'normH',normH,...
             'Id_bar',Id_bar, 'normId', normId, 'r',r, 'd',d, 'c',c, 'sij',sij,'usize',usize, ...
             'delta',delta(ite),'m_dual',m_dual, 'l_bbox',l_bbox, 'u_bbox',u_bbox,'qq',qq);

    %Solve the SDP feasibility problem (10).
    [flag(ite), xx_opt(:,:,ite)] = SDPAlg(data);

    if flag(ite)==0
        deltal=delta(ite);
    elseif flag(ite)==1
        deltau=delta(ite);
    end
end

%find the optimal transformation matrix.
flagindex = find(flag==0);
feasind = max(flagindex);
transformW = xx_opt(:,:,feasind);

end