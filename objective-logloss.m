'CROSS-ENTROPY (LOGLOSS) COST';

function ctx = logloss_init(eps=1e-8)
    ctx.unit = 'logloss';
    ctx.eps  = eps;
end;

function E = logloss_cost(ctx, Z, Y)
    m = count(Z);
    E = Y .* log(Z);
    E = nansum(E, 2);
    E = -sum(E) / m;
end

function dE = logloss_gradient(ctx, Z, Y)
    eps = ctx.eps;
    dE  = -Y ./ (Z + eps);
end

function acc = logloss_accuracy(ctx, Z, Y)
    Z   = maxhots(Z);
    eq  = Z == Y;
    eq  = all(eq, 1);
    n1  = nnz(eq);
    m   = count(eq);
    acc = n1 / m;
end
