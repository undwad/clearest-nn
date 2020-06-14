'RECURRENT LAYER';

function ctx = recurrent_init(size_x, size_z)
    size_z     = tosize(size_z);
    nx         = size_x(1);
    nz         = size_z(1);
    [Sz,Sx]    = stack_transform_matricies(nz, nx);
    ctx.unit   = 'recurrent';
    ctx.Gz     = model(nz+nx, {'dense', nz}, 'tanh');
    ctx.Sz     = Sz;
    ctx.Sx     = Sx;
    ctx.size_x = size_x;    
    ctx.size_z = size_z;
    ctx.num_p  = ctx.Gz.num_p;
end

function z = recurrent_predict(ctx, X)
    [~, z] = recurrent_forward(ctx, X);
end

function [ctx, z] = recurrent_forward(ctx, X)
    scope(ctx);
    m   = count(X);
    nz  = size_z(1);
    Z   = zeros(nz, m);
    z   = zeros(nz, 1);
    XXX = cell(1, m);
    for i = 1:m
        Z(:,i) = z;
        x      = X(:,i);
        zx     = Sz*z + Sx*x; 
        [Gz,z] = forward(Gz, zx);
        XXX{i} = Gz.XXX;
    end
    ctx.Z   = Z;
    ctx.XXX = XXX;
end

function [ctx, _E_X, gg] = recurrent_backward(ctx, X, z, _E_z)
    scope(ctx);
    m    = count(X);
    gg   = zeros(Gz.num_p, 1);
    _E_X = zeros(size(X));
    for i = m:-1:1
        Gz.XXX        = XXX{i};
        [~,ggg,_E_zx] = backward(Gz, z, _E_z);
        gg           += ggg;
        _E_x          = Sx' * _E_zx;
        _E_z          = Sz' * _E_zx;
        _E_X(:,i)     = _E_x;
        z             = Z(:,i);
    end
end

function ctx = recurrent_update(ctx, gg)
    scope(ctx);
    n      = Gz.num_p;
    ctx.Gz = update(Gz, gg(1:n));
end

function pp = recurrent_export(ctx)
    pp = export(ctx.Gz);
end

function ctx = recurrent_import(ctx, pp)
    ctx.Gz = import(ctx.Gz, pp);
end
