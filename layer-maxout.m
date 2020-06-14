'MAXOUT LAYER';

function ctx = maxout_init(size_x, size_z, num_u=2)
    size_z     = tosize(size_z);
    nx         = size_x(1);
    nz         = size_z(1);
    ctx.unit   = 'maxout';
    ctx.W      = glorot(num_u, nx, nz);
    ctx.b      = zeros(num_u, nz);
    ctx.size_x = size_x;    
    ctx.size_z = size_z;
    ctx.num_u  = num_u;
    ctx.num_p  = numel(ctx.W) + numel(ctx.b);
end

function Z = maxout_predict(ctx, X)
    [~, Z] = maxout_forward(ctx, X);
end

function [ctx, Z] = maxout_forward(ctx, X)
    scope(ctx);
    m        = count(X);
    [nz, ~]  = split(size_z);
    Z        = zeros(nz, m);
    ctx.mask = zeros(num_u, m, nz);
    for i = 1:nz
        Wi              = W(:, :, i);
        bi              = b(:, i);
        Zi              = Wi * X + bi;
        zi              = max(Zi, [], 1);
        Z(i, :)         = zi;
        mask            = Zi == zi;
        ctx.mask(:,:,i) = mask;
    end
end

function [ctx, dEx, gg] = maxout_backward(ctx, X, Z, dEz)
    scope(ctx);
    [nx, m] = size(X);
    [nz, ~] = split(size_z);
    d.W     = zeros(num_u, nx, nz);
    d.b     = zeros(num_u, nz);
    dEx     = zeros(nx, m); 
    for i = 1:nz
        mask       = ctx.mask(:, :, i);
        Wi         = W(:, :, i);
        bi         = b(:, i);
        dEzi       = dEz(i, :);
        dEzi       = dEzi .* mask;
        dWi        = (dEzi * X') / m;
        dbi        = sum(dEzi, dim=2) / m;
        dExi       = Wi' * dEzi;
        d.W(:,:,i) = dWi;
        d.b(:,i)   = dbi;
        dEx       += dExi;
    end
    gg = maxout_export(d);
end

function ctx = maxout_update(ctx, gg)
    ctx = dense_update(ctx, gg);
end

function pp = maxout_export(ctx)
    pp = dense_export(ctx);
end

function ctx = maxout_import(ctx, pp)
    ctx = dense_import(ctx, pp);
end
