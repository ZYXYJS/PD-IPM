% ====================== the obj function value for one step PDHG is similar to the LP problem ============================
function [x_f,y_f]=onestep_PDHG(A,b,c,x,s,y, tau, sigma, tol)

m = size(A,1);
n = size(A,2);
mu   = (x.'*s)/n;
stop = false;
iter   = 0;
itermax = 1000;
while stop == false && iter < itermax
    

z = x+ tau*(A.'*y - c);
x_f = 0.5*(z + sqrt(z.^2 + 4*tau*mu));
% snew = c - A.'*y - (1/tau)*(xnew - xold);

ynew = y- sigma*(A*(2.*x_f-x)-b);

mu    = 0.5.*mu;

r_p =  A.'*y-c -mu./x;
r_d =  A*x-b;
% r_ps = x_f.*s_f-mu;
residual = [r_d; r_p];

x = x_f;
% s = s_f;
y = ynew;

if norm(residual) < tol
    stop = true;
    x_f   = x;
    % s_f   = s;
    y_f   = y;
end

iter = iter+1;   
end

if iter == itermax
    x_f   = x;
    % s_f   = s;
    y_f   = y;
end

end


% function [x_f,s_f,y_f]= PDHG_IPM (A,b,c,x,s,y, tau, sigma, tol)

% m = size(A,1);
% n = size(A,2);
% mu   = (x.'*s)/n;
% stop = false;
% iter   = 0;
% itermax = 1000;
% while stop == false && iter < itermax
    
% % ===================== obj fucntion value not the same ==============================================
% % [x_f, s_f] = PDHG_INNER_NEWTON_IPM(A, b, c, x, s, y, tau, sigma);
% % =============================================================================================

% % =====================  occures with negative component =======================================
% % [x_f, s_f] = PDHG_INNER_GLOBAL_IPM(A, b, c, x, s, y, tau, sigma);
% % ========================================================================================



% % PDHG Updtate    
% % z = x + tau.*(A.'*y) - tau.*c;
% % xnew = (z+ sqrt(z.^2 + 4*mu*tau))/2;

% % xbar  = x +tau.*(A.'*y); 
% % Fhandle = @(xs) F(xs,c,tau, xbar,mu);
% % xtest      = fsolve(Fhandle,[xnew; max(A.'*y-c,0)] );

% % test_res= norm(xnew-xtest(1:n),2);
% % fprintf('X-XTEST = %.6e\n', test_res);



% ynew = y- sigma*(A*(2.*x_f-x)-b);

% mu    = 0.5.*mu;

% r_p =  A.'*y-c -s;
% r_d =  A*x-b;
% r_ps = x_f.*s_f-mu;
% residual = [r_d; r_p; r_ps];

% x = x_f;
% s = s_f;
% y = ynew;

% if norm(residual) < tol
%     stop = true;
%     x_f   = x;
%     s_f   = s;
%     y_f   = y;
% end

% iter = iter+1;   
% end

% if iter == itermax
%     x_f   = x;
%     s_f   = s;
%     y_f   = y;
% end

% end





function [x_f, s_f] = PDHG_INNER_NEWTON_IPM(A, b, c, x, s, y, tau, sigma)

m = size(A,1);
n = size(A,2);

mu   = (x.'*s)/n;
stop = false;
iter = 0;
inner_max = 500;

while ~stop && iter < inner_max
    
    iter = iter + 1;

    e = ones(n,1);
    X = diag(x);
    S = diag(s);

    au_x = s + A.'*y - c;
    au_s = mu*e - x.*s;
    au   = [au_x; au_s];

    Jac_t = [(1/tau)*eye(n), -eye(n)];
    Jac_b = [S, X];
    Jab   = [Jac_t; Jac_b];

    % XS_result = Jab \ au;
    XS_result  = linsolve(Jab, au);

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

    alpha = 0.8 * alpha;
    beta  = 0.8 * beta;

    xnew = x + 0.9*alpha * delta_x;
    snew = s + 0.9*beta  * delta_s;

    C_AY_s = norm(c - A.'*y - s,2) ;
    XS     = x.*s - mu*e;
    xs_norm = norm(XS,2);

    fprintf('Iter %d: XSnorm = %.3e, CAYS = %.3e\n', iter, xs_norm, C_AY_s);

    if C_AY_s < 1e-5 && xs_norm < 1e-7
        stop = true;
    elseif min(xnew) < 0 || min(snew) < 0
        warning('Negative component detected, stopping.');
        stop = true;
    end

    x = xnew;
    s = snew;

end
    x_f = x;
    s_f = s;
end






function [x_f, s_f] = PDHG_INNER_GLOBAL_IPM(A, b, c, x, s, y, tau, sigma)

m = size(A,1);
n = size(A,2);

mu   = (x.'*s)/n;
stop = false;
iter = 0;
inner_max = 1;
xold = x;
while ~stop && iter < inner_max
    e = ones(n,1);
    iter = iter + 1;
    % xbar  = x +tau.*(A.'*y); 
    % Fhandle = @(xs) F(xs,xold,c,tau, A,y,mu);
    % xs = [x;s];
    % XS_result = fsolve(Fhandle, xs);
    % xnew = XS_result(1:n);
    % snew = XS_result(n+1:end);

    z = xold + tau*(A.'*y - c);
    xnew = 0.5*(z + sqrt(z.^2 + 4*tau*mu));
    snew = c - A.'*y - (1/tau)*(xnew - xold);


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



function res = F(xs, xold, c, tau, A, y, mu)
n = size(c,1);
xnew = xs(1:n,1);
snew = xs(n+1:end,1);

bb = c + (1/tau)*(xnew - xold) - A.'*y - snew;

cc = xnew .* snew - mu;

res = [bb; cc];
end


