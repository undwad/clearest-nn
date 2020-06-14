'L2 REGULARIZATION OPTIMIZER';

function ctx = L2_regularization_init(num_p, rate)
    ctx.unit = 'L2_regularization';
    ctx.rate = rate;
end;

function E = L2_regularization_cost(ctx, ppp, E, m)
    rate = ctx.rate / m;
    R    = sum(ppp.^2) * rate/2;
    E    = E + R;
end

function [ctx, ggg] = L2_regularization_optimize(ctx, ppp, ggg, m)
    rate = ctx.rate / m;
    rrr  = ppp .* rate;
    ggg  = ggg .+ rrr;
end
