'MOMENTUM OPTIMIZER';

function ctx = momentum_init(num_p, rate, mu=0.9)
    ctx.unit = 'momentum';
    ctx.rate = rate;
    ctx.mu   = mu;
    ctx.ggg_ = zeros(num_p, 1);
end;

function [ctx, ggg] = momentum_optimize(ctx, ppp, ggg, m)
    scope(ctx);
    ggg  = ctx.ggg_ = mu.*ggg_ .+ rate.*ggg;
end
