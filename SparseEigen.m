function [sp_vectors, vectors, values] = SparseEigen(C, d, rho_nrm, q, V)

% INPUT
%   C :             n-by-m data matrix (n samples, m variables).
%   d :             1-by-q vector with weights.
%   rho_nrm :       Sparsity weight factor. Values from 0 to 1.
%   q :             Number of estimated eigenvectors.
%   V (optional):   m-by-q initial point matrix. If not provided the eigenvectors 
%                   of the sample covariance matrix are used.
%
% OUTPUT
%   sp_vectors :    m-by-q matrix, columns corresponding to leading
%                   sparse eigenvectors.
%   vectors :       m-by-q matrix, columns corresponding to leading
%                   eigenvectors.
%   values :        q-by-1 vector corresponding to the leading eigenvalues.
%
% INFO  
%   Reference:      K. Benidis, Y. Sun, P. Babu, D.P. Palomar "Orthogonal Sparse 
%                   PCA and Covariance Estimation via Procrustes Reformulation"
%                   IEEE Transactions on Signal Processing, vol 64, Dec. 2016.
%
%   Algorithm:      This algorithm corresponds to the accelerated IMRP algorithm 
%                   of the referenced paper. 
%                    
%   Link :          http://www.danielppalomar.com/publications.html


% Initialize
[n, m] = size(C); 
k = 0;
maxIter = 1000;

% Preallocation
V_tld = zeros(m,q);
H = zeros(m,q);
F_v = zeros(maxIter,1); % record the objective at each iteration
g = zeros(m,q);

% Rho
[Uc, Sc, Vc] = svd(C,'econ');
Sc2 = diag(Sc).^2;
rho = rho_nrm*max(sum(C.^2)).*(Sc2(1:q)./Sc2(1))'.*d;

% Initial Point
if nargin < 5
    V = Vc(:,1:q);
end

% Decreasing epsilon, p
K = 10;
p1 = 1; % first value of p
pT = 7; % last value of p
gamma = (pT/p1)^(1/K);  
pp = p1*gamma.^(0:K);
pp = 10.^(-pp);

tol = pp*1e-2; % tolerance for convergence
Eps = pp; % epsilon

for ee = 1:K+1
    p = pp(ee);
    epsi = Eps(ee);
    c1 = log(1+1/p);
    c2 = 2*(p + epsi)*c1;
    w0 = (1/(epsi*c2))*ones(m*q,1);
    flg = 1;
    
    while 1
    k = k + 1;  
    
    %-------------------------------------%
    % First iteration of the acceleration %
    
    % weights
    w = w0;
    ind = (abs(V(:)) > epsi);
    w(ind) = (0.5/c1)./(V(ind).^2 + p*abs(V(ind)));
    
    % MM
    for i = 1:q
        w_tmp = w((i-1)*m+1:i*m);
        V_tld(:,i) = V(:,i).*d(i);
        H(:,i) = (w_tmp - max(w_tmp)*ones(m,1)).*V(:,i)*rho(i);
    end

    G = Vc*((Vc'*V_tld).*Sc2(:,ones(1,q)));

    % update
    [V_l,S_B,V_r] = svd(G - H,'econ');
    V1 = V_l*V_r';

    %--------------------------------------%
    % Second iteration of the acceleration %

    % weights
    w = w0;
    ind = (abs(V1(:)) > epsi);
    w(ind) = (0.5/c1)./(V1(ind).^2 + p*abs(V1(ind)));

    % MM
    for i = 1:q
        w_tmp = w((i-1)*m+1:i*m);
        V_tld(:,i) = V1(:,i).*d(i);
        H(:,i) = (w_tmp - max(w_tmp)*ones(m,1)).*V1(:,i)*rho(i);
    end

    G = Vc*((Vc'*V_tld).*Sc2(:,ones(1,q)));
    
    % update
    [V_l,S_B,V_r] = svd(G - H,'econ');
    V2 = V_l*V_r';
   
    %--------------%
    % Acceleration %

    R = V1 - V;
    U = V2 - V1 - R;           
    a = min(-norm(R,'fro')/norm(U,'fro'),-1);

    while 1 % backtracking loop
        V0 = V - 2*a.*R + a^2.*U;

        % Projection
        [V_l,S_B,V_r] = svd(V0,'econ');
        V0 = V_l*V_r';

        g(abs(V0)<=epsi) = V0(abs(V0)<=epsi).^2/(epsi*c2);
        g(abs(V0)>epsi) = log((p + abs(V0(abs(V0)>epsi)))/(p + epsi))/c1 + epsi/c2;

        F_v(k) = sum((C*V0).^2)*d' - (sum(g)*rho');

        if flg == 0 && (F_v(k)*(1 + sign(F_v(k))*1e-9)) <= F_v(max(k-1,1))
            a = (a-1)/2;
        else
            V = V0;
            break
        end
    end
    
    % Stopping criterion
    if flg == 0
        rel_change = abs(F_v(k) - F_v(k-1))/max(1,abs(F_v(k-1))); % relative change in objective
        if rel_change <= tol(ee) || k >= maxIter
            F_v = F_v(1:k);
            break
        end
    end
    flg = 0;
    end
end

V(abs(V) < 1e-10) = 0; % threshold
nrm = 1./sqrt(sum(V.^2));
V = nrm(ones(m,1),:).*V;        

sp_vectors = V;
vectors = Vc(:,1:q);
values = Sc2(1:q)./n;

        
