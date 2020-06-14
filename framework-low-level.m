'LOW-LEVEL FRAMEWORK';

function ctx = init_ctx(cfg, varargin)
    _cfg_ = cfg;
    if !isa(cfg, 'cell')
        cfg = {cfg}; 
    end
    unit = cfg{1};
    init = sprintf('%s_init', unit); 
    ctx  = feval(init, varargin{:}, cfg{2:end});  
    if !isfield(ctx, 'unit')
        ctx.unit = unit;
    end
    ctx.cfg = _cfg_;
end

function CTX = model(size_x, varargin)
    size_x     = tosize(size_x);
    CTX        = struct();
    CTX.layers = {};
    CTX.num_p  = 0;
    CTX.size_x = size_x;
    n          = length(varargin);
    for i = 1:n
        ctx            = init_ctx(varargin{i}, size_x);
        CTX.layers{i}  = ctx;
        CTX.num_p      = CTX.num_p + ctx.num_p;
        size_x         = ctx.size_z;
    end
    
    CTX.size_z = size_x;
end

function CTX = optimization(CTX, varargin)
    CTX.optimizers = {};
    n = length(varargin);
    for i = 1:n
        ctx               = init_ctx(varargin{i}, CTX.num_p);        
        CTX.optimizers{i} = ctx;
    end
end

function CTX = objective(CTX, cfg)
    ctx           = init_ctx(cfg);        
    CTX.objective = ctx;
end

function ppp = export(CTX)
    ppp = [];
    n   = length(CTX.layers);
    for i = 1:n
        ctx = CTX.layers{i};
        fun = [ctx.unit '_export'];
        if ctx.num_p > 0
            pp  = feval(fun, ctx);
            ppp = [ ppp; pp ];
        end
    end
end

function CTX = import(CTX, ppp)
    p = 0;
    n = length(CTX.layers);
    for i = 1:n
        ctx = CTX.layers{i};
        fun = [ctx.unit '_import'];
        np  = ctx.num_p;
        if np > 0
            pp            = ppp(p+1:p+np);
            CTX.layers{i} = feval(fun, ctx, pp);
            p             = p + np;
        end
    end
end

function Z = predict(CTX, X)
    n = length(CTX.layers);
    for i = 1:n
        ctx = CTX.layers{i};
        fun = [ctx.unit '_predict'];
        Z   = feval(fun, ctx, X);
        X   = Z;
    end
end

function [CTX, Z] = forward(CTX, X)
    XXX = {};
    n   = length(CTX.layers);
    for i = 1:n
        XXX{i}        = X;
        ctx           = CTX.layers{i};
        fun           = [ctx.unit '_forward'];
        [ctx, Z]      = feval(fun, ctx, X);
        CTX.layers{i} = ctx;
        X             = Z;
    end
    CTX.XXX = XXX;
end

function E = cost(CTX, Z, Y)
    ctx = CTX.objective;
    fun = [ctx.unit '_cost'];
    E   = feval(fun, ctx, Z, Y);
end

function dE = gradient(CTX, Z, Y)
    ctx = CTX.objective;
    fun = [ctx.unit '_gradient'];
    dE  = feval(fun, ctx, Z, Y);
end

function acc = accuracy(CTX, Z, Y)
    acc = nan;
    ctx = CTX.objective;
    fun = [ctx.unit '_accuracy'];
    if exist(fun)
        acc = feval(fun, ctx, Z, Y);
    end
end

function [CTX, ggg, dE] = backward(CTX, Z, dE)
    ggg = [];
    m   = count(Z);
    n   = length(CTX.layers);
    for i = n:-1:1
        ctx           = CTX.layers{i};
        fun           = [ctx.unit '_backward'];
        X             = CTX.XXX{i};  
        [ctx, dE, gg] = feval(fun, ctx, X, Z, dE);
        CTX.layers{i} = ctx;
        ggg           = [ gg; ggg ];
        Z             = X;
    end
end

function E = optimize_cost(CTX, E, m)
    ppp = export(CTX);
    n   = length(CTX.optimizers);
    for i = 1:n
        ctx = CTX.optimizers{i};
        fun = [ctx.unit '_cost'];
        if exist(fun)
            E = feval(fun, ctx, ppp, E, m);
        end
    end
end

function [CTX, ggg] = optimize_gradient(CTX, ggg, m)
    ppp = export(CTX);
    n   = length(CTX.optimizers);
    for i = 1:n
        ctx               = CTX.optimizers{i};
        fun               = [ctx.unit '_optimize'];
        [ctx, ggg]        = feval(fun, ctx, ppp, ggg, m);
        CTX.optimizers{i} = ctx;
    end
end

function CTX = update(CTX, ggg)
    p = 0;
    n = length(CTX.layers);
    for i = 1:n
        ctx = CTX.layers{i};
        fun = [ctx.unit '_update'];
        np  = ctx.num_p;
        if np > 0
            gg            = ggg(p+1:p+np);
            CTX.layers{i} = feval(fun, ctx, gg);
            p             = p + np;
        end
    end
end

function [diff, ddd, k] = gradient_check(CTX1, ggg1, X, Y, timeout=5, eps=1e-5)
    t    = tic();
    m    = count(X);
    ppp1 = export(CTX1);
    ggg2 = zeros(size(ggg1));
    ppp2 = ppp1;
    n    = length(ppp1);
    iii  = randperm(n);
    for k = 1:n
        i = iii(k); 
    
        ppp2(i) = ppp1(i) - eps;
        CTX2    = import(CTX1, ppp2);
        [~,Z]   = forward(CTX2, X);
        E_minus = cost(CTX2, Z, Y); 
        
        ppp2(i) = ppp1(i) + eps;
        CTX2    = import(CTX1, ppp2);
        [~,Z]   = forward(CTX2, X);
        E_plus  = cost(CTX2, Z, Y); 
       
        ggg2(i) = (E_plus - E_minus) / (eps + eps);
        
        ppp2(i) = ppp1(i);
        
        if toc(t) > timeout
            break;
        end
    end
    
    ggg1(iii(k+1:end)) = 0;
    
    ddd  = abs(ggg1 - ggg2);
    diff = norm(ddd) / (norm(ggg1) + norm(ggg2));
end

function [ctx, i, unit] = getunit(ctxs, unit)
    if iscell(unit)
        [unit,idx] = deal(unit{1}, unit{2});
    else
        idx = 1;
    end  
    n = count(ctxs);
    for i = 1:n
        ctx = ctxs{i};
        if strcmp(ctx.unit, unit)
            idx -= 1;
            if idx == 0
                return;
            end
        end
    end
    ctx = [];
end

function ctx = updatefields(ctx, varargin)
    m = fix(count(varargin) / 2);
    for i = 0:m-1
        key       = varargin{i*2 + 1};
        val       = varargin{i*2 + 2};
        ctx.(key) = val;
    end
end

function ctxs = updateunit(ctxs, unit, varargin)
    [ctx, i] = getunit(ctxs, unit);
    if !isempty(ctx)
        ctx     = updatefields(ctx, varargin{:});
        ctxs{i} = ctx;
    end
end

function unit = assignunit(ctxs, unit)
    [ctx,~,unit] = getunit(ctxs, unit);
    if !isempty(ctx)
        assignin('caller', unit, ctx);
    end
end

function CTX = tune(CTX, unit, varargin)
    CTX.layers     = updateunit(CTX.layers,     unit, varargin{:});
    CTX.optimizers = updateunit(CTX.optimizers, unit, varargin{:});    
end
