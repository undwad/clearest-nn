'ADAM OPTIMIZER';

function ctx = adam_init(num_p, rate, beta1=0.9, beta2=0.999, eps=1e-8)
    ctx.unit  = 'adam';
    ctx.rate  = rate;
    ctx.beta1 = beta1;
    ctx.beta2 = beta2;
    ctx.eps   = eps;
    ctx.ggg1  = zeros(num_p, 1);
    ctx.ggg2  = zeros(num_p, 1);
    ctx.times = 0;
end;

function [ctx, ggg] = adam_optimize(ctx, ppp, ggg, m)
    scope(ctx);
    times = ctx.times = times + 1;
    ggg1  = ctx.ggg1  = beta1 .* ggg1 .+ (1 - beta1) .* ggg;
    ggg2  = ctx.ggg2  = beta2 .* ggg2 .+ (1 - beta2) .* (ggg.^2);
    corr1 = ggg1 ./ (1 - beta1^times);
    corr2 = ggg2 ./ (1 - beta2^times);
    ggg   = corr1 ./ (sqrt(corr2) .+ eps) .* rate;
end
