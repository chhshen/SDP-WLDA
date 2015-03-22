
function [flag,xx_opt] = SDPAlg(data)
%Function of solving the SDP feasibility problem (10) by L-BFGS-B.
%
% Full description can be found in: 
%   "Worst-Case Linear Discriminant Analysis as Scalable Semidefinite Feasibility Problems." 
%    Hui Li, Chunhua Shen, Anton van den Hengel and Qinfeng Shi. TIP 2015
%
% USAGE
%  [flag,xx_opt] = SDPAlg_sp2(data)

% INPUTS
%   data     - a struct variable including variables needed by L-BFGS-B and other functions.
%   algorithm.

% OUTPUTS
%   flag   - shows the feasibility of the problem (flag=1:feasible;
%   flag=0:infeasible).
%   xx_opt - the corresponding transformation matrix.


%   disp('-----------------------------------------------');
    disp('SDP feasibility problem solver_fast start ...');   

    d  = data.d;     % The dimensionality of training data in original space 
    u_init=1*ones(data.m_dual,1);  % the initial values for Lagrangial dual variables used in L-BFGS-B
        
    % Solve the Lagrangian dual problem (13) by L-BFGS-B algorithm.
    [u_opt, A_posit_opt, flag] = solve_dual_lbfgsb(u_init, data);
     X = A_posit_opt;   
     Z = X(1:d,1:d);
 
    % recover transformation matrix xx_opt from Z = xx_opt * xx_opt'
    [VB,DB]=eigs(Z,data.r); 
    xx_opt=VB;  

    disp('wlda_solve_fast end');
    disp('-----------------------------------------------');
end

function [u_opt, A_posit_opt,flag] = solve_dual_lbfgsb(u_init, data)
% Function of Solving the Lagrangian dual problem (13) by L-BFGS-B algorithm.

    fcn = @(u) calc_dual_obj_grad_lbfgsb(u, data);
    lbfgsb_opts = struct('x0',     u_init, ...
                         'maxIts', 10000);
    %Calculate the optimal dual variable u_opt using L-BFGS-B.
    [u_opt, ~, info] = lbfgsb(fcn, data.l_bbox, data.u_bbox, lbfgsb_opts);
    %Check the feasibility condition and compute the positive part of A_bar by eigen-decomposition. 
    [A_posit_opt,flag] = calc_a_posit(u_opt, data);
    
end

function [A_posit, flag] = calc_a_posit(u, data)
% Function of calculating the positive part of A_bar (A_posit), and checking
% the feability condition as well.
    flag=0;    
    usize=data.usize;    % the total number of dual variables.  
    
    A = calc_a(u, data);  % calculate A_bar by dual variables.
    [A_posit, A_minus] = calc_pos_neg_part_sp(A);  % calculate the positive and negative parts of A_bar respectively.

    %Check the feasibility condition.
    checka=norm(A_posit,'fro');
    checkb=[data.r;data.qq];
    c=data.c;
    checkd=abs(u((usize+1):(data.m_dual))'*checkb);
    checktest=u((usize+1):(data.m_dual))'*checkb;
    check=checka/checkd;            
    if (check < 1e-3) && (checktest>0)
   %     disp('infeasible');
        flag=1;
    end
          
end

function [obj, grad] = calc_dual_obj_grad_lbfgsb(u, data)
%Function of calculating the objective and gradient of the objective
%function in Lagrangian dual problem (13).
   global flag;
   [A_posit, flag] = calc_a_posit(u, data);
   
   % Calculate the objective values
   obj  = calc_dual_obj(u, A_posit, data);

   % Calculate the gradient values
   grad = calc_dual_grad(data, A_posit);

end
      
function  A = calc_a(u, data)
%Function of computing A_bar basd on dual variables.
    norterm=[data.normS; data.normId; data.normH];
    u=u./ norterm;

    usize = data.usize;   
    S_b=data.S_b;
    S_w=data.S_w;
    Id_bar=data.Id_bar;
    d=data.d;
    c=data.c;

    hh=0.5*d*(d+1);
    for bet=1:data.sij
        tpu(bet,1)=sum(u(c*(bet-1)+1:c*bet));
    end
    
    A_temp1=S_b*tpu;
    A_up1=reshape(A_temp1,d,d);
    
    for bek=1:c
        uc=u(bek:c:usize);
        tpk(bek,1)=sum(uc);
    end

    A_temp2=S_w*tpk;
    A_up2=reshape(A_temp2,d,d);   
    A1=A_up1-data.delta*A_up2;    
    A1=A1+u(usize+1)*Id_bar;
    A=[A1 zeros(d);zeros(d) zeros(d)];
    ii=usize+1;
    
    a=ones(d,d);
    b=tril(a);
    idx=find(b~=0);
    c(idx)=u(ii+1:ii+hh);
    uu=reshape(c,d,d);
    utem=uu+uu'-diag(diag(uu));
    
    ucal(1:d,1:d)=utem;
    ucal((d+1):2*d,(d+1):2*d)=utem;   
    A=A+ucal;

end

function obj = calc_dual_obj(u, A_posit, data)
% Function of calculating the objective values
    d=data.d;
    dip=data.usize+1;
    m_dual=data.m_dual;

    tempP=u(dip+1:m_dual)./data.normH;
    tempb=find(tril(ones(d)));
    tempc=zeros(d);
    tempc(tempb)=tempP;
    P_nor=tempc';

    obj = (0.5) * sum(A_posit(:) .^ 2) - u(dip)*data.r / data.normId - trace(P_nor);

end

function grad = calc_dual_grad(data, A_posit)
% Function of calculating the gradient values   
    m_dual=data.m_dual;
    d=data.d;
    dip=data.usize;
    grad = zeros(m_dual, 1); 
    S_b=data.S_b;
    S_w=data.S_w;
    Aup=A_posit(1:d,1:d);
    
    G1l=S_b'*Aup(:);
    G1r=S_w'*Aup(:);
    
    ii=0;
    for bet=1:data.sij
        for bek=1:data.c
            ii=ii+1;
            grad(ii,1)=(G1l(bet)-data.delta*G1r(bek))/data.normS(ii);
        end
    end
      
    Id_bw=[data.Id_bar, zeros(d);zeros(d) zeros(d)];
    Id_bw_nor = Id_bw / data.normId; 

    grad(dip+1,1)=A_posit(:)' * Id_bw_nor(:) - data.r/data.normId;
    grad(dip+2:m_dual,1)=data.H_bb'*A_posit(:)- data.qq./data.normH;
end
    
function [X_plus, X_minus] = calc_pos_neg_part_sp(X)

% eigenvectors and eigenvalues
    [V, D] = eig(X);

dd = real( diag(D) );
idxs_minus = find(dd < 0);
n_minus = length(idxs_minus);
D_minus = sparse(1:n_minus, 1:n_minus, dd(idxs_minus), n_minus, n_minus, n_minus); 
V_minus = V(:, idxs_minus);

% fprintf('rank of negative part: %d\n', n_minus);

% output X_minus and X_plus
X_minus = (V_minus * D_minus) * V_minus';
X_plus = X - X_minus;

end
