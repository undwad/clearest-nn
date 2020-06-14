'POOLING 2D LAYER';

function ctx = pool2D_init(size_x, size_z, range_f, mode)
    assert(mode == 'max' || mode =='avg');

    size_z(3) = size_x(3);
    ctx = convinit('pool2D', size_x, size_z, range_f);
    if any(ctx.size_z != size_z)
        warning(['pool2D output size changed from ' mat2str(size_z) ' to ' mat2str(ctx.size_z)]);
    end
    ctx.mode  = mode;
    ctx.num_p = 0;
end

function Z = pool2D_predict(ctx, X)
    scope(ctx);
    [zh, zw, xc] = split(size_z);
    [fh, fw, xc] = split(size_f);
    m            = count(X);
    Z            = zeros(zh, zw, xc, m);
    [X_, HH, WW] = zeropad(X, padding);
    for h = 1:zh
        for w = 1:zw
            [hh, ww] = convslice(size_f, stride, h, w);
            x        = X_(hh, ww, :, :);
            if mode == 'avg'
                z = mean(mean(x, 1), 2);
            else
                z = max(max(x, [], 1), [], 2);
            end
            z = reshape(z, 1, xc, m);
            Z(h, w, :, :) = z;
        end
    end        
end

function [ctx, Z] = pool2D_forward(ctx, X)
    Z = pool2D_predict(ctx, X);
end

function [ctx, dE, gg] = pool2D_backward(ctx, X, Z, dEz)
    scope(ctx);
    [zh, zw, xc] = split(size_z);
    [fh, fw, xc] = split(size_f);
    m            = count(X);
    Z            = zeros(zh, zw, xc, m);
    [X_, HH, WW] = zeropad(X, padding);
    dEx          = zeros(size(X_));
    for h = 1:zh
        for w = 1:zw
            [hh, ww] = convslice(size_f, stride, h, w);
            x        = X_(hh, ww, :, :);
            z        = Z(h, w, :, :);
            de       = dEz(h, w, :, :);
            if mode == 'avg'
                mask = ones(fh, fw, xc) / (fh*fw); 
            else
                mask = x == z;
            end
            dEx(hh, ww, :, :) += de .* mask;         
        end
    end        
    if padding > 0
        dE = dEx(HH, WW, :, :);
    else
        dE = dEx;
    end
    gg = [];
end
