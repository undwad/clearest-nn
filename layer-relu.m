'RELU LAYER';

function ctx = relu_init(size_x)
    ctx.unit   = 'relu';
    ctx.size_x = size_x;    
    ctx.size_z = size_x;
    ctx.num_p  = 0;    
end

function Z = relu_predict(ctx, X)
    Z = (X > 0) .* X;
end

function [ctx, Z] = relu_forward(ctx, X)
     Z = relu_predict(ctx, X);
end

function [ctx, dE, gg] = relu_backward(ctx, X, Z, dE) 
    dZ = X > 0;
    dE = dE .* dZ;
    gg = []; 
end
