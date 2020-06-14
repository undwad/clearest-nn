'DENSE (AFFINE) LAYER';

function ctx = dense_init(size_x, size_z)
    size_z     = tosize(size_z);
    nx         = size_x(1);
    nz         = size_z(1);
    ctx.unit   = 'dense';
    ctx.W      = glorot(nz, nx);
    ctx.b      = zeros(nz, 1);
    ctx.size_x = size_x;    
    ctx.size_z = size_z;
    ctx.num_p  = numel(ctx.W) + numel(ctx.b);
end

function Z = dense_predict(ctx, X)
    scope(ctx);
    Z = W * X + b;
end

function [ctx, Z] = dense_forward(ctx, X)
    Z = dense_predict(ctx, X);
end

function [ctx, dE, gg] = dense_backward(ctx, X, Z, dE)
    scope(ctx);
    m   = count(X);
    d.W = (dE * X')      / m;
    d.b = sum(dE, dim=2) / m;
    dE  = W' * dE;
    gg  = dense_export(d);
end

function ctx = dense_update(ctx, gg)
    d      = ctx;    
    d      = dense_import(d, gg);
    ctx.W -= d.W;
    ctx.b -= d.b;
end

function pp = dense_export(ctx)
    scope(ctx);
    pp = [ W(:); b(:) ];
end

function ctx = dense_import(ctx, pp)
    scope(ctx);
    n     = numel(W);
    ctx.W = reshape(pp(1:n),     size(W));
    ctx.b = reshape(pp(n+1:end), size(b));
end
