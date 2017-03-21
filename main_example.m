% Description :     Main example of sparse eigenvector extraction. The function 
%                   SparseEigen corresponds to the accelerated IMRP algorithm of
%                   the referenced paper.
%
% Reference:        K. Benidis, Y. Sun, P. Babu, D.P. Palomar "Orthogonal Sparse 
%                   PCA and Covariance Estimation via Procrustes Reformulation"
%                   IEEE Transactions on Signal Processing, vol 64, Dec. 2016.
%                    
% Link :            http://www.danielppalomar.com/publications.html

clear all
close all

%% Parameters

m = 500;  % dimension
n = 100; % number of samples
q = 3; % number of sparse eigenvectors to be estimated
SpCard = 0.2*m; % cardinality of the sparse eigenvectors
d = ones(1,q); % IMRP parameter
        
%% True Covariance 

% Sparse eigenvectors
V = randn(m,m);
tmp = zeros(m,q);
for i = 1:max(q,2)
    ind1 = (i-1)*SpCard + 1;
    ind2 = i*SpCard;
    tmp(ind1:ind2,i) = 1/sqrt(SpCard);
    V(:,i) = tmp(:,i);
end

[V,Q] = qr(V); % orthogonalization, but keep the first eigenvectors to be same as v

% Eigenvalues
vl = ones(m,1); 
for i = 1:q
    vl(i) = 100*(q+1-i);
end

% Covariance matrix
R = V*diag(vl)*V'; 
    
%% Data Matrix

C = mvnrnd(zeros(1,m),R,n); % random data with underlying sparse structure
C = C - repmat(mean(C),n,1); % center the data

%% Sparse Eigenvector Extraction 
              
rho = 0.6; % normalized sparsity inducing parameter
[Res.sp_vectors, Res.vectors, Res.values] = SparseEigen(C, d, rho, q);
      
%% Plots 

Rec(:,1) = abs(diag(Res.sp_vectors'*V(:,1:q))) % recovery

figure(1) 
subplot(311)
bar(Res.sp_vectors(:,1).*sign(Res.sp_vectors(1,1)))
hold on
plot(V(:,1).*sign(V(1,1)), 'r')
xlabel('Index')
title('First Eigenvector')
grid on

subplot(312)
bar(Res.sp_vectors(:,2).*sign(Res.sp_vectors(SpCard+1,2)))
hold on
plot(V(:,2).*sign(V(SpCard+1,2)), 'r')
xlabel('Index')
title('Second Eigenvector')
grid on

subplot(313)
bar(Res.sp_vectors(:,3).*sign(Res.sp_vectors(2*SpCard+1,3)))
hold on
plot(V(:,3).*sign(V(2*SpCard+1,3)), 'r')
xlabel('Index')
title('Third Eigenvector')
grid on

