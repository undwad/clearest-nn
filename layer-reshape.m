'RESHAPE LAYER';

function ctx = reshape_init(size_x, size_z=[])
    flatten     = isempty(size_z);
    if flatten
        size_z  = [prod(size_x), 1];
    end
    size_z      = tosize(size_z);
    ctx.unit    = 'reshape';
    ctx.size_x  = size_x;    
    ctx.size_z  = size_z;
    ctx.flatten = flatten;
    ctx.num_p   = 0;    
end

function Z = reshape_predict(ctx, X)
    scope(ctx);
    m = count(X);  
    if flatten
        size_z = size_z(1);
    end
    Z = reshape(X, [size_z, m]);
end

function [ctx, Z] = reshape_forward(ctx, X)
    scope(ctx);
    Z = reshape_predict(ctx, X);
end

function [ctx, dE, gg] = reshape_backward(ctx, X, Z, dE) 
    scope(ctx);
    m  = count(dE);
    dE = reshape(dE, [size_x, m]);
    gg = []; 
end
