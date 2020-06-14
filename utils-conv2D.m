'CONVOLUTIONAL UTILS';

function n = convoutdim(n, f, p, s)
    n = (n + 2*p - f) / s + 1;
end

function size_z = convoutsize(size_x, size_f, p, s)
    [xh, xw] = split(size_x);
    [fh, fw] = split(size_f);
    zh       = convoutdim(xh, fh, p, s);
    zw       = convoutdim(xw, fw, p, s);
    size_z   = [zh, zw];
end

function [size_z_, size_f, padding, stride] = convapproxparams(size_x, size_z, range_f)
    minerr = inf;
    for f = range_f
        for p = 0:f-1
            for s = 1:f
                sz = convoutsize(size_x, [f,f], p, s);
                err  = norm(sz - size_z);
                if err < minerr && !any(sz - fix(sz))
                      size_z_ = sz;
                      size_f  = [f,f];
                      padding = p;
                      stride  = s;
                      minerr  = err;
                end
            end
        end
    end
end

function [hh, ww] = padslice(size_x, p)
    [xh, xw] = split(size_x);
    yh = p + xh + p;
    yw = p + xw + p;
    hh = p+1:p+xh;
    ww = p+1:p+xw;
end

function [hh, ww] = convslice(size_f, s, h, w)
    [fh, fw] = split(size_f);
    vert1    = (h-1) * s  + 1;
    vertN    = vert1 + fh - 1;
    horz1    = (w-1) * s  + 1;
    horzN    = horz1 + fw - 1;
    hh       = vert1:vertN;
    ww       = horz1:horzN;
end

function ctx = convinit(unit, size_x, size_z, range_f)
    [xh, xw, xc]           = split(size_x);
    [zh, zw, zc]           = split(size_z);
    [size_z, size_f, p, s] = convapproxparams([xh, xw], [zh, zw], range_f);
    size_z(3)              = zc;
    size_f(3:4)            = [xc, zc]; 
    ctx.unit    = unit;
    ctx.size_x  = size_x;    
    ctx.size_z  = size_z;
    ctx.size_f  = size_f;
    ctx.padding = p;
    ctx.stride  = s;
end

function [X_, HH, WW] = zeropad(X, p)
    [xh, xw, xc, m]      = size(X);
    if p > 0
        [HH, WW]         = padslice([xh, xw], p);
        X_               = zeros(p+xh+p, p+xw+p, xc, m);
        X_(HH, WW, :, :) = X;
    else
        X_               = X;
        HH = WW          = 0;
    end
end
