function [w,fvalues,gnorm] = LevenbergMarquardt(r_and_J,w,kmax,tol)
Delta_max = 1.2; % the max trust-region radius
Delta_min = 1e-12; % the minimal trust-region radius
R = 0.3; % the initial radius
eta = 0.01; % step rejection parameter
rho_good = 0.75;
rho_bad = 0.25;


gnorm = zeros(kmax,1);
fvalues = zeros(kmax,1);
[r,J]= r_and_J(w);
f = 0.5*norm(r)*norm(r);
g = J'*r;


k=1;
normg = 10;
while k<kmax && normg>tol
    B = J'*J + (1e-6)*eye(size(J'*J));
    f = 0.5*norm(r)*norm(r);
    g = J'*r;
    pstar = -B\g; % unconstrained minimizer
    if norm(pstar) <= R
        p = pstar;
    else % solve constrained minimization problem
        lam = 1; % initial guess for lambda
        while 1
            
            B1 = B + lam*eye(size(B));
            C = chol(B1); % do Cholesky factorization of B
            p = -C\(C'\g); % solve B1*p = -g
            np = norm(p);
            dd = abs(np - R); % R is the trust region radius
            if dd < 1e-5
                break
            end
            q = C'\p; % solve C^\top q = p
            nq = norm(q);
            lamnew = lam + (np/nq)^2*(np - R)/R;
            if lamnew < 0
                lam = 0.5*lam;
            else
                lam = lamnew;
            end
        end
    end
    w = w + p;
    m = norm(J*p+r);
    [r,J] = r_and_J(w);
    fnew = 0.5*norm(r)*norm(r);
    gnew = J'*r;
    % mnew = f + g'*p + 0.5*p'*B*p;
    m = f + g'*p + 0.5*p'*B*p;
    rho = (f - fnew)/(f - m);
    % adjust the trust region
    if rho < rho_bad
        R = max([0.25*R,Delta_min]);
    else
        if rho > rho_good && norm(pstar) <= R
            R = min([Delta_max,2*R]);
        end
    end
    
    if rho > eta  
        f = fnew;
        g = gnew;
        normg = norm(g);
        gnorm(k) = normg;
        fvalues(k) = f;
        % disp(normg)
        % fprintf('Accepted\n');
    else
        w = w-p;
        % fprintf('Rejected \n');
        disp(norm(gnew))
    end
    k=k+1;
end

end

%%
function f = qloss(I,Xtrain,label,w,lam)
f = sum(log(1 + exp(-myquadratic(Xtrain,label,I,w))))/length(I) + 0.5*lam*w'*w;
end
%%
function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end
%%
function q = myquadratic(Xtrain,label,I,w)
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end



