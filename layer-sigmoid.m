'SIGMOID LAYER';

function Z = sigmoid(X)
    Z = 1 ./ (1 .+ e.^(-X));
end

function ctx = sigmoid_init(size_x)
    ctx.unit   = 'sigmoid';
    ctx.size_x = size_x;    
    ctx.size_z = size_x;
    ctx.num_p  = 0;    
end

function Z = sigmoid_predict(ctx, X)
    Z = sigmoid(X);
end

function [ctx, Z] = sigmoid_forward(ctx, X)
    Z = sigmoid_predict(ctx, X);
end

function [ctx, dE, gg] = sigmoid_backward(ctx, X, Z, dE) 
    dZ = Z .* (1 - Z);
    dE = dE .* dZ;
    gg = []; 
end
