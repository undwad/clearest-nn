'DESCENT OPTIMIZER';

function ctx = descent_init(num_p, rate)
    ctx.unit = 'descent';
    ctx.rate = rate;
end;

function [ctx, ggg] = descent_optimize(ctx, ppp, ggg, m)
    ggg = ggg .* ctx.rate;
end
