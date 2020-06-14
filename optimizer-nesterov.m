'NESTEROV MOMENTUM OPTIMIZER';

function ctx = nesterov_init(num_p, rate, mu=0.9)
    ctx.unit = 'nesterov';
    ctx.rate = rate;
    ctx.mu   = mu;
    ctx.ggg1 = zeros(num_p, 1);
    ctx.ggg2 = zeros(num_p, 1);
end;

function [ctx, ggg] = nesterov_optimize(ctx, ppp, ggg, m)
    scope(ctx);
    ctx.ggg1 = ggg1 = ggg2;
    ctx.ggg2 = ggg2 = mu.*ggg2 .+ rate.*ggg;
    ggg  = mu.*ggg1 .+ (1-mu).*ggg2;
end
