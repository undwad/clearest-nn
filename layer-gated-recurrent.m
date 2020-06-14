'GATED RECURRENT LAYER';

function ctx = gated_recurrent_init(size_x, size_z, eps=1e-8)
    size_z     = tosize(size_z);
    nx         = size_x(1);
    nz         = size_z(1);
    [Sz,Sx]    = stack_transform_matricies(nz, nx);
    ctx.unit   = 'gated_recurrent';
    ctx.Gu     = model(nz+nx, {'dense', nz}, 'sigmoid');
    ctx.Gr     = model(nz+nx, {'dense', nz}, 'sigmoid');
    ctx.Gc     = model(nz+nx, {'dense', nz}, 'tanh');
    ctx.Sz     = Sz;
    ctx.Sx     = Sx;
    ctx.size_x = size_x;    
    ctx.size_z = size_z;
    ctx.num_p  = ctx.Gr.num_p + ctx.Gu.num_p + ctx.Gc.num_p;
    ctx.eps    = eps;
end

function z = gated_recurrent_predict(ctx, X)
    [~, z] = gated_recurrent_forward(ctx, X);
end

function [ctx, z] = gated_recurrent_forward(ctx, X)
    scope(ctx);
    [nx,m]  = size(X);
    nz      = size_z(1);
    Z       = zeros(nz, m);
    z       = zeros(nz, 1);
    TMP     = cell(1, m);
    for i   = 1:m
        x      = X(:,i);
        z0     = Z(:,i) = z;
        z0x    = Sz*z0 + Sx*x;      % z0x(z0,x) = [z0;x]
        [Gu,u] = forward(Gu, z0x);  % u(z0x) 
        [Gr,r] = forward(Gr, z0x);  % r(z0x) 
        z1x    = Sz*(r.*z0) + Sx*x; % z1x(r,z0,x) = [r.*z0;x]
        [Gc,c] = forward(Gc, z1x);  % c(z1x)  
        z      = (1-u).*c + u.*z0;  % z(u,c,z0), u(z0), c(z0)    
        TMP{i} = fromscope('r','u','c','Gu.XXX','Gr.XXX','Gc.XXX');
    end
    ctx.Z   = Z;
    ctx.TMP = TMP;
end

function [ctx, _E_X, gg] = gated_recurrent_backward(ctx, X, z, _E_z)
    scope(ctx);
    [nx,m]  = size(X);
    [nz,~]  = size(_E_z);
    d.Gu    = zeros(Gu.num_p, 1); 
    d.Gr    = zeros(Gr.num_p, 1); 
    d.Gc    = zeros(Gc.num_p, 1); 
    _E_X    = zeros(nx, m);
    for i   = m:-1:1
        scope(TMP{i});
        x  = X(:,i);
        z0 = Z(:,i);

        _z_u            = z0 - c;
        _E_u            = _E_z .* _z_u;
        Gu.XXX          = Gu_XXX;   
        [~,ggg,_Eu_z0x] = backward(Gu, u, _E_u);
        d.Gu           += ggg;
        _u_z0           = Sz' * _Eu_z0x ./ (_E_u+eps);

        _z_c            = 1 - u;
        _E_c            = _E_z .* _z_c;
        Gc.XXX          = Gc_XXX;   
        [~,ggg,_Ec_z1x] = backward(Gc, c, _E_c);
        d.Gc           += ggg;
        _c_z0_          = (Sz' * _Ec_z1x) .* r  ./ (_E_c+eps); 
        _c_r            = (Sz' * _Ec_z1x) .* z0 ./ (_E_c+eps);

        _z_r            = _z_c .* _c_r;
        _E_r            = _E_z .* _z_r;
        Gr.XXX          = Gr_XXX;   
        [~,ggg,_Er_z0x] = backward(Gr, r, _E_r);
        d.Gr           += ggg;
        _r_z0           = Sz' * _Er_z0x ./ (_E_r+eps);

        _z_z0_ = u;
        _c_z0  = _c_r.*_r_z0 + _c_z0_;
        _z_z0  = _z_c.*_c_z0 + _z_u.*_u_z0 + _z_z0_;
        _E_z0  = _E_z .* _z_z0;    
        _E_z   = _E_z0;
    end
    gg = [ d.Gu; d.Gr; d.Gc ];
end

function ctx = gated_recurrent_update(ctx, gg)
    scope(ctx);
    p  = 1; n = Gu.num_p; ctx.Gu = update(Gu, gg(p:p+n-1));
    p += n; n = Gr.num_p; ctx.Gr = update(Gr, gg(p:p+n-1));
    p += n; n = Gc.num_p; ctx.Gc = update(Gc, gg(p:p+n-1));
end

function pp = gated_recurrent_export(ctx)
    scope(ctx);
    pp = [ export(Gu); export(Gr); export(Gc) ];
end

function ctx = gated_recurrent_import(ctx, pp)
    scope(ctx);
    p  = 1; n = Gu.num_p; ctx.Gu = import(Gu, pp(p:p+n-1));
    p += n; n = Gr.num_p; ctx.Gr = import(Gr, pp(p:p+n-1));
    p += n; n = Gc.num_p; ctx.Gc = import(Gc, pp(p:p+n-1));
end
