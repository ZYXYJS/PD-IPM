function [x_f,s_f,y_f]=PDHG_IPM(A,b,c,x,s,y, tau,sigma, tol)

m = size(A,1);
n = size(A,2);

mu   = (x.'*s)/n;
stop = false;
iter   = 0;
itermax = 1000;

while stop == false && iter < itermax

% PDHG Updtate    
z = x + tau.*(A.'*y) - tau.*c;
xnew = (z+ sqrt(z.^2 + 4*mu*tau))/2;

xbar  = x +tau.*(A.'*y); 
Fhandle = @(xs) F(xs,c,tau, xbar,mu);
xtest      = fsolve(Fhandle,[x; max(A.'*y-c,0)] );

test_res= norm(xnew-xtest(1:n),2);
fprintf('X-XTEST = %.6e\n', test_res);

ynew = y- sigma*(A*(2.*xnew-x)-b);

mu    = 0.9.*mu;

residual = [A*x-b; A.'*y-c - max(A.'*y-c,0)];

x = xnew;
y = ynew;

if norm(residual) < tol
    stop = true;
    x_f   = x;
    s_f   =  max(A.'*y-c,0);
    y_f   = y;
end

 iter = iter+1;   
end

if iter == itermax
    x_f   = x;
    s_f   = max(A.'*y-c,0);
    y_f   = y;
end


end


function  res = F(xs,c,tau, xbar,mu)

n               = size(c,1);
x               = xs(1:n,1);
s               = xs(n+1:end,1);

bb = c + (1/tau).*x - (1/tau).*xbar -s;
cc =  x.*s - mu;

res            =   [bb; cc];  


end
