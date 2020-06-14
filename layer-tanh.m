'TANH LAYER';

function ctx = tanh_init(size_x)
    ctx.unit   = 'tanh';
    ctx.size_x = size_x;    
    ctx.size_z = size_x;
    ctx.num_p  = 0;    
end

function Z = tanh_predict(ctx, X)
    Z = tanh(X);
end

function [ctx, Z] = tanh_forward(ctx, X)
    Z = tanh_predict(ctx, X);
end

function [ctx, dE, gg] = tanh_backward(ctx, X, Z, dE) 
    dZ = 1 - Z.^2;
    dE = dE .* dZ;
    gg = []; 
end
