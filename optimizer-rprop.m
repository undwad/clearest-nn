'RPROP OPTIMIZER';

function ctx = rprop_init(num_p, rate0=1, etapos=1.2, etaneg=0.5)
    ctx.unit   = 'rprop';
    ctx.signs  = zeros(num_p, 1);
    ctx.rates  = ones(num_p, 1) .* rate0;
    ctx.etapos = etapos;
    ctx.etaneg = etaneg;
end;

function [ctx, ggg] = rprop_optimize(ctx, ppp, ggg, m)
    scope(ctx);
    signs     = sign(ggg);
    samesigns = ctx.signs == signs;
    diffsigns = ~samesigns;
    factors   = samesigns.*etapos .+ diffsigns.*etaneg;
    ctx.rates = ctx.rates .* factors;
    ctx.signs = signs;
    ggg       = signs .* ctx.rates;
end
