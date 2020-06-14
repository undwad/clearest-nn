'GRADIENT CLIPPING OPTIMIZER';

function ctx = gradient_clipping_init(num_p, maxnorm)
    ctx.unit    = 'gradient_clipping';
    ctx.maxnorm = maxnorm;
end;

function [ctx, ggg] = gradient_clipping_optimize(ctx, ppp, ggg, m)
    maxnorm = ctx.maxnorm;
    norm    = ctx.norm = norm(ggg);
    if norm > maxnorm
        ggg = ggg .* (maxnorm / norm);
    end
end
