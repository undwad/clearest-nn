'MULTITASK CROSS-ENTROPY (LOGLOSS) COST';

function ctx = multitaskloss_init(eps=1e-8)
    ctx.unit = 'multitaskloss';
    ctx.eps  = eps;
end;

function E = multitaskloss_cost(ctx, Z, Y)
    E = 0;
    n = rows(Y);
    for i = 1:n
        Z_ = Z(i, :);
        Y_ = Y(i, :);
        E += binlogloss_cost(ctx, Z_, Y_);
    end
end

function dE = multitaskloss_gradient(ctx, Z, Y)
    [n,m] = size(Y);
    dE    = zeros(n,m);
    for i = 1:n
        Z_      = Z(i, :);
        Y_      = Y(i, :);
        dE_     = binlogloss_gradient(ctx, Z_, Y_);
        dE(i,:) = dE_;
    end
end

function acc = multitaskloss_accuracy(ctx, Z, Y)
    acc = binlogloss_accuracy(ctx, Z, Y);
end
