'DROPOUT LAYER';

function ctx = dropout_init(size_x, keep_prob)
    ctx.unit      = 'dropout';
    ctx.size_x    = size_x;    
    ctx.size_z    = size_x;
    ctx.num_p     = 0;    
    ctx.keep_prob = keep_prob;
end

function Z = dropout_predict(ctx, X)
    Z = X;
end

function [ctx, Z] = dropout_forward(ctx, X)
    scope(ctx);
    m        = count(X);
    sz       = setcount(size_z, m);
    ctx.mask = (rand(sz) < keep_prob) ./ keep_prob;
    Z        = X .* ctx.mask;
end

function [ctx, dE, gg] = dropout_backward(ctx, X, Z, dE) 
    scope(ctx);
    dE = dE .* mask;
    gg = []; 
end
