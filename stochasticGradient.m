function [w,fvalues,gnorm] = stochasticGradient(w,kmax,tol,fun,gfun,bsz)

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
while norm_g > tol && k < 1200
        curr_ind = Ig(b:b+5);
        % Ig = randperm(n,20);
        g = gfun(curr_ind,w);
        norm_g = norm(g);
        gnorm(k) = norm_g;
        fvalues(k) = fun(curr_ind,w);
        disp(norm_g)
        w = w - g*(step/2^k);
        k = k+1;
        b=b+5;
end

end


