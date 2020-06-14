'SOFTMAX LAYER';

function ctx = softmax_init(size_x, eps=1e-8)
    ctx.unit   = 'softmax';
    ctx.size_x = size_x;    
    ctx.size_z = size_x;
    ctx.num_p  = 0;   
    ctx.eps    = eps;
end

function Z = softmax(X, eps=1e-8)
    X  = X - max(X);
    eX = e.^X; 
    Z  = eX ./ (sum(eX) + eps);
end

function [x,idx] = softmaxpick(X, eps=1e-8)
    Z   = softmax(X, eps);
    oh  = mnrnd(1, Z);
    idx = find(oh);
    x   = X(idx, :);
end

function Z = softmax_predict(ctx, X)
    Z = softmax(X, ctx.eps);
end

function [ctx, Z] = softmax_forward(ctx, X)
    Z = softmax_predict(ctx, X);
end

function [ctx, dE, gg] = softmax_backward(ctx, X, Z, dE)
    m = count(Z);
    for i = 1:m
        z        = Z(:, i);
        DZ       = diag(z) - z * z';
        dE(:, i) = DZ * dE(:, i);
    end
    gg = []; 
end
