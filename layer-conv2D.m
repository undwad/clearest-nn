'CONVOLUTIONAL 2D LAYER';

function ctx = conv2D_init(size_x, size_z, range_f)
    ctx = convinit('conv2D', size_x, size_z, range_f);
    if any(ctx.size_z != size_z)
        warning(['conv2D output size changed from ' mat2str(size_z) ' to ' mat2str(ctx.size_z)]);
    end
    zc        = ctx.size_z(3);
    ctx.F     = glorot(ctx.size_f);
    ctx.b     = zeros(1, zc);
    ctx.num_p = numel(ctx.F) + numel(ctx.b);
end

function Z = conv2D_predict(ctx, X)
    scope(ctx);
    [zh, zw, zc]     = split(size_z);
    [fh, fw, xc, zc] = split(size_f);
    m                = count(X);
    Z                = zeros(zh, zw, zc, m);
    [X_, HH, WW]     = zeropad(X, padding);
    for h = 1:zh
        for w = 1:zw
            for c = 1:zc
                f             = F(:, :, :, c);
                [hh, ww]      = convslice(size_f, stride, h, w);
                x             = X_(hh, ww, :, :);
                z             = x .* f;
                z             = sum(sum(sum(z, 1), 2), 3);
                z             = reshape(z, 1, m);
                z             = z .+ b(c);
                Z(h, w, c, :) = z;
            end
        end
    end        
end

function [ctx, Z] = conv2D_forward(ctx, X)
    Z = conv2D_predict(ctx, X);
end

function [ctx, dE, gg] = conv2D_backward(ctx, X, Z, dEz)
    scope(ctx);
    [zh, zw, zc]     = split(size_z);
    [fh, fw, xc, zc] = split(size_f);
    m                = count(X);
    Z                = zeros(zh, zw, xc, m);
    [X_, HH, WW]     = zeropad(X, padding);
    d.F              = zeros(size(F));
    d.b              = zeros(size(b));
    dEx              = zeros(size(X_));
    n                = zh * zw * m;
    for h = 1:zh
        for w = 1:zw
            for c = 1:zc
                f                  = F(:, :, :, c);
                [hh, ww]           = convslice(size_f, stride, h, w);
                x                  = X_(hh, ww, :, :);
                de                 = dEz(h, w, c, :);
                d.F(:, :, :, c)   += sum(de .* x, dim=4) / n;
                d.b(c)            += sum(de, dim=4)      / n;
                dEx(hh, ww, :, :) += de .* f;         
            end
        end
    end        
    if padding > 0
        dE = dEx(HH, WW, :, :);
    else
        dE = dEx;
    end
    gg = conv2D_export(d);
end

function ctx = conv2D_update(ctx, gg)
    d      = ctx;    
    d      = conv2D_import(d, gg);
    ctx.F -= d.F;
    ctx.b -= d.b;
end

function pp = conv2D_export(ctx)
    scope(ctx);
    pp = [ F(:); b(:) ];
end

function ctx = conv2D_import(ctx, pp)
    scope(ctx);
    n     = numel(F);
    ctx.F = reshape(pp(1:n),     size(F));
    ctx.b = reshape(pp(n+1:end), size(b));
end
