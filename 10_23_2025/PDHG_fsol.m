function [x_f,s_f,y_f]= PDHG_fsol(A,b,c,x,s,y, tau, sigma, tol)

m = size(A,1);
n = size(A,2);
mu   = (x.'*s)/n;
stop = false;
iter   = 0;
itermax = 1000;
while stop == false && iter < itermax
    
% ===================== obj fucntion value not the same ==============================================
% [x_f, s_f] = PDHG_INNER_NEWTON_IPM(A, b, c, x, s, y, tau, sigma,mu);
% =============================================================================================

% =====================  occures with negative component =======================================
[x_f, s_f] = PDHG_INNER_GLOBAL_IPM(A, b, c, x, s, y, tau, sigma);
% ========================================================================================



% PDHG Updtate    
% z = x + tau.*(A.'*y) - tau.*c;
% xnew = (z+ sqrt(z.^2 + 4*mu*tau))/2;

% xbar  = x +tau.*(A.'*y); 
% Fhandle = @(xs) F(xs,c,tau, xbar,mu);
% xtest      = fsolve(Fhandle,[xnew; max(A.'*y-c,0)] );

% test_res= norm(xnew-xtest(1:n),2);
% fprintf('X-XTEST = %.6e\n', test_res);



ynew = y- sigma*(A*(2.*x_f-x)-b);

mu    = 0.5.*mu;
% mu = compute_mu(A,c,x,y,s,n,tau,mu)

% r_p =  A.'*y-c -s;
% r_d =  A*x-b;
% r_ps = x.*s-mu;

r_p  = norm(A*x- b) ;
r_d  = norm(c-A.'*y - s) ;
% fprintf('SF=%.6e\n', norm(s_f));
% fprintf('ATC=%.6e\n', norm(A.'*y - c));

r_ps  = norm(x.* s - mu);

residual = [r_d; r_p; r_ps];
fprintf('residuals: Rp=%.3e, Rd=%.3e, Rc=%.3e\n', r_p, r_d, r_ps);
fprintf('res_norm=%.6e\n', norm(residual));
if norm(residual) < tol
    stop = true;
    x_f   = x;
    s_f   = s;
    y_f   = y;
end

% fprintf('res = %.6e\n', norm(residual));
x = x_f;
s = s_f;
y = ynew;
iter = iter+1;  

if iter == itermax
    x_f   = x;
    s_f   = s;
    y_f   = y;
end



end


end



function [x_f, s_f] = PDHG_INNER_GLOBAL_IPM(A, b, c, x, s, y, tau, sigma)

m = size(A,1);
n = size(A,2);

mu   = (x.'*s)/n;
stop = false;
iter = 0;
inner_max = 100;
xold = x;
while ~stop && iter < inner_max
    e = ones(n,1);
    iter = iter + 1;
    xbar  = x +tau.*(A.'*y); 
    z = log(x);
    options = optimoptions('fsolve', 'Display', 'off');
    Fhandle = @(zs) F(zs,xold,c,tau, A,y,mu);
    zs = [z;s];
    XS_result = fsolve(Fhandle, zs,options);
    znew = XS_result(1:n);
    snew = XS_result(n+1:end);
    xnew = exp(znew);

% =================  the solution of fsolve should be the same in the following ========================== 
% ===================  but since it involves to solve a QP problem, it has two solution, and fsolve dont know which one to choose ======================
% ==================== so may be we need other nonlinear function to consider the inner loop ===================================
%     % z = xold + tau*(A.'*y - c);
%     % xnew = 0.5*(z + sqrt(z.^2 + 4*tau*mu));
%     % snew = c - A.'*y - (1/tau)*(xnew - xold);
% ============================================================================================

    C_AY_s = norm(c - A.'*y -s,2);
    XS     = x.*s - mu*e;
    xs_norm = norm(XS,2);

    fprintf('Iter %d: XSnorm = %.3e, CAYS = %.3e\n', iter, xs_norm, C_AY_s);

    if C_AY_s < 1e-5 && xs_norm < 1e-7
        stop = true;
    elseif min(xnew) < 0 || min(snew) < 0
        warning('Negative component detected, stopping.');
        stop = true;
    end

    xold = x;
    x = xnew;
    s = snew;

end
    x_f = x;
    s_f = s;
end



function res = F(zs, xold, c, tau, A, y, mu)
n = size(c,1);
znew = zs(1:n,1);
snew = zs(n+1:end,1);
xnew = exp(znew);
bb = c + (1/tau)*(xnew - xold) - A.'*y - snew;

cc = xnew .* snew - mu;

res = [bb; cc];
end



function mu_new = compute_mu(A,c,x,y,s,n,tau,mu)
    alpha0 = 0.9;
    beta0 = 0.9;
    X = diag(x(:));
    S = diag(s(:));

    au_x = s + A' * y - c;
    au_s = -x .* s;
    au = [au_x; au_s];

    Jac_t = [(1/tau) * ones(n, n), -eye(n)];
    Jac_b = [S, X];
    Jab = [Jac_t; Jac_b];

   
    % reg_lambda = 1e-8;
    % Jab_reg = Jab + reg_lambda * eye(2 * n);

    XS_result = Jab \ au;

    delta_x = XS_result(1:n);
    delta_s = XS_result(n+1:end);

   
    if any(delta_x < 0)
        alpha = min(1, min(-x(delta_x < 0) ./ delta_x(delta_x < 0)));
    else
        alpha = 1.0;
    end

   
    if any(delta_s < 0)
        beta = min(1, min(-s(delta_s < 0) ./ delta_s(delta_s < 0)));
    else
        beta = 1.0;
    end

   
    xnew = x + alpha0 * alpha * delta_x;
    snew= s + beta0 * beta * delta_s;


    mu_new = mean(xnew .* snew);
    theta = (mu_new/mu)^3;
    mu_new = theta*mu_new;
    fprintf('theta = %.7f\n', theta);
   
end