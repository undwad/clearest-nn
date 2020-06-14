'BINARY CROSS-ENTROPY (LOGLOSS) COST';

function ctx = binlogloss_init(eps=1e-8)
    ctx.unit = 'binlogloss';
    ctx.eps  = eps;
end;

function E = binlogloss_cost(ctx, Z, Y)
    m = count(Z);
    E0 = nansum((1 - Y) .* log(1 - Z));
    E1 = nansum((    Y) .* log(    Z));
    E  = -sum(E0 + E1) / m;   
end

function dE = binlogloss_gradient(ctx, Z, Y)
    eps = ctx.eps;
    dE  = (1 - Y)./(1 - Z + eps) - Y./(Z + eps);
end

function acc = binlogloss_accuracy(ctx, Z, Y)
    Z   = Z > 0.5;
    eq  = Z == Y;
    eq  = all(eq, 1);
    n1  = nnz(eq);
    m   = count(eq);
    acc = n1 / m;
end
