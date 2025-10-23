clc;
clear;
m = 100;
n = 300;

seed = 1;
rng(seed);

A = rand(m,n);
b = A*rand(n,1);
c = rand(n,1);

a    = norm(A.'*A,2);

tau     = 0.8/a; 
sigma = 0.8/a;
tol       = 1e-4;



% [x_f,s_f,y_f] = PDHG_IPM_new(A,b,c,ones(n,1),ones(n,1),ones(m,1), tau,sigma, tol);
% fprintf('obj = %.6e\n', c.'*x_f);

[x_f,y_f] = onestep_PDHG(A,b,c,ones(n,1),ones(n,1),ones(m,1), tau,sigma, tol);
fprintf('obj = %.6e\n', c.'*x_f);

%[x_f,s_f,y_f]= PDHG_fsol(A,b,c,ones(n,1),ones(n,1),ones(m,1), tau, sigma, tol);
%fprintf('obj = %.6e\n', c.'*x_f);

xlp = linprog(c,[],[],A,b,zeros(n,1),[]);
fprintf('obj2 = %.6e\n', c.'*xlp);
