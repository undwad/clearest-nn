'OPTIMIZATION STATISTICS';

function ctx = stats_init(num_p)
    ctx.unit = 'stats';
    ctx.iter = 0;
end;

function [ctx, ggg] = stats_optimize(ctx, ppp, ggg, m)
    scope(ctx);
    gnorm     = norm(ggg);
    pnorm     = norm(ppp);
    ctx.ratio = gnorm / pnorm;
    ctx.iter  = iter + 1; 
end
