function [w,fvalues,gnorm] = Nesterov(w,kmax,tol,fun,gfun,bsz)



gnorm = zeros(kmax,1);
fvalues = zeros(kmax,1);

n =13007;
I= 1:n;
g = gfun(I,w);
norm_g = norm(g);
% gnorm(1) = norm_g;
k=1;
step = 0.2;
Ig = randperm(n);
b=1;
y = w;
lambda_old = 0;

while norm_g > tol && k < 1200
        curr_ind = Ig(b:b+5);

        g = gfun(curr_ind,w);
        ynew = w - step*g;
        lambda = 0.5*(sqrt(1 + 4 * lambda_old^2)+1);
        gamma = (1-lambda_old) / lambda;
        w = (1-gamma) * y + gamma * y;
        norm_g = norm(g);
        gnorm(k) = norm_g;
        fvalues(k) = fun(curr_ind,w);
        disp(norm_g)
        w = w - g*(step/2^k);
        k = k+1;
        b=b+5;
        lambda_old=lambda;
        y = ynew;
end





end