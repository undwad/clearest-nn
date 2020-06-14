'UTILS';

function eq = isapprox(x, y, eps=1e-7)
    eq = abs(x - y) < eps;
    eq = all(eq(:));
end

function sz = tosize(sz)
    if isscalar(sz)
        sz = [ sz, 1 ];        
    end
end

function n = count(X)
    n = size(X)(end);
end

function sz = setcount(sz, n)
    sz(end) = n;
end

function dim = dimN(X)
    dim = count(size(X));
end

function X = flattenelems(X)
    X = reshape(X, [], count(X));
end

function enc = onehot(val, all)
    n        = length(all);
    idx      = find(all == val);
    enc      = zeros(n, 1);
    enc(idx) = 1;
end

function val = onecold(enc, all, alt)
    idx = find(enc == 1);
    val = all(idx);
    if isempty(val)
        val = alt;
    end
end

function encs = onehots(vals, all)
    n    = length(all);
    m    = count(vals);
    encs = zeros(n, m);
    for i = 1:m
        val        = vals(i);
        encs(:, i) = onehot(val, all);
    end
end

function vals = onecolds(encs, all, alt)
    m    = count(encs);
    vals = zeros(1, m);
    for i = 1:m
        enc     = encs(:, i);
        vals(i) = onecold(enc, all, alt);
    end
end

function encs = maxhots(out)
    n      = rows(out);
    [~, i] = max(out);
    encs   = onehots(i, 1:n);
end

function vals = maxcolds(out, all, alt)
    encs = maxhots(out);
    vals = onecolds(encs, all, alt);  
end

function [X, code] = slice(X, varargin)
    if iscell(X)
        [ open, close ] = deal('{', '}');
    else
        [ open, close ] = deal('(', ')');
    end
    sz  = size(X);
    n   = count(sz);
    tmp = not(cellfun(@isempty, varargin));
    m   = count(tmp);
    assert(nnz(tmp) == m-1);
    colons = @(i) strjoin(num2cell(repmat(':', 1, n-m+1)), ',');
    index  = @(i) sprintf('varargin{%d}', i);
    tocode = @(i) ifelse(isempty(varargin{i}), colons(i), index(i));
    args = arrayfun(tocode, 1:m, 'UniformOutput', 0);
    args = strjoin(args, ',');
    code = ['X' open args close];
    X    = eval(code);
end

function scope(ctx)
    for [v, k] = ctx
        k = strrep(k, '.', '_');
        assignin('caller', k, v);
    end
end

function s = fromscope(varargin)
    n = count(varargin);
    for i = 1:n
        var     = varargin{i};    
        s.(var) = evalin('caller', var);
    end
end

function varargout = shuffle(varargin)
    m   = count(varargin{1});
    idx = randperm(m);
    for i = 1:nargout
        varargout{i} = slice(varargin{i}, [], idx);
    end
end

function varargout = pick(varargin)
    m   = count(varargin{1});
    idx = randi(m);
    for i = 1:nargout
        varargout{i} = slice(varargin{i}, [], idx);
    end
end

function Y = map(f, X)
    m = count(X);
    Y = cell(1, m);
    for i = 1:m
        x    = slice(X, [], i);
        y    = f(x);
        Y{i} = y;
    end
    if !iscell(X)
        Y = horzcat(Y{:});
    end
end

function varargout = split(X)
    xxx = X(:);
    for i = 1:nargout
        varargout{i} = xxx(i);
    end
end

function X = glorot(varargin)
    sz = [varargin{:}];
    d  = sqrt(6 / (sum(sz) - 1));
    X  = unifrnd(-d, d, sz);
end

function [X, avg, var] = normalize(X, eps=1e-7)
    dim = dimN(X);
    avg = mean(X, dim);
    var = var(X, opt=1, dim);
    X   = X .- avg; 
    X   = X ./ sqrt(var+eps); 
end

function X = padright(X, x, n)
    m  = length(X);
    xx = repmat(x, 1, n-m);
    X  = [X xx];
end

function X = padleft(X, x, n)
    m  = length(X);
    xx = repmat(x, 1, n-m);
    X  = [xx X];
end

function [Sz, Sx] = stack_transform_matricies(nz, nx)
    Sz = shift(diag(ones(nz,1),nz+nx,nz), 0,dim=1);
    Sx = shift(diag(ones(nx,1),nz+nx,nx),nz,dim=1);
end

function xy = mergereplace(x, y)
    xy = cell2struct([struct2cell(x);struct2cell(y)],[fieldnames(x);fieldnames(y)]);
end

function x = setsome(x, y, varargin)
    x(varargin{:}) = y;
end
