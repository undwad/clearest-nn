'BATCH NORM LAYER';

function ctx = batchnorm_init(size_x, eps=1e-8)
    ctx.unit     = 'batchnorm';
    ctx.gam      = glorot(size_x);
    ctx.bet      = glorot(size_x);
    ctx.size_x   = size_x;    
    ctx.size_z   = size_x;
    ctx.num_p    = numel(ctx.gam) + numel(ctx.bet);
    ctx.eps      = eps;
    ctx.cma_mu   = zeros(size_x);
    ctx.cma_sig2 = zeros(size_x);
    ctx.num_iter = 0;   
end

function Z = batchnorm_predict(ctx, X)
    scope(ctx);
    Xhat = (X-cma_mu) ./ sqrt(cma_sig2+eps); # n⨯m
    Z    = Xhat.*gam + bet;                  # n⨯m
end

function [ctx, Z] = batchnorm_forward(ctx, X)   # n⨯m
    scope(ctx);
    sum  = @(x) sum(x, dim=2);
    m    = count(X);
    mu   = sum(X) / m;               # n⨯1       
    sig2 = sum((X-mu).^2) / m;       # n⨯1
    Xhat = (X-mu) ./ sqrt(sig2+eps); # n⨯m
    Z    = Xhat.*gam + bet;          # n⨯m

    ctx.cma_mu   = (num_iter*cma_mu   + mu)   / (num_iter+1);
    ctx.cma_sig2 = (num_iter*cma_sig2 + sig2) / (num_iter+1);
    ctx.num_iter = num_iter + 1;
    
    ctx = mergereplace(ctx, fromscope('mu','sig2','Xhat'));
end

function [ctx, dE, gg] = batchnorm_backward(ctx, X, Z, dE)             # n⨯m
    scope(ctx);
    sum        = @(x) sum(x, dim=2);
    n          = size_x(1);
    m          = count(X);    
    _E_Z       = dE;                                                   # n⨯m
    _E_bet     = sum(_E_Z);                                            # n⨯1            
    _E_gam     = sum(_E_Z .* Xhat);                                    # n⨯1                
    _E_Xhat    = _E_Z .* gam;                                          # n⨯m
%   _Xhat_sig2 = -1/2 * sum(X-mu) ./ (sig2+eps).^(3/2);                # n⨯1
    _Xhat_mu   = -1 * ones(n,1) ./ sqrt(sig2+eps);                     # n⨯1
    _Xhat_X    = ones(n,m) ./ sqrt(sig2+eps);                          # n⨯m 
    _sig2_X    = 2/m * (X-mu);                                         # n⨯m
    _mu_X      = ones(n,m) / m;                                        # n⨯m               
%   _E_sig2    = sum(_E_Xhat) .* _Xhat_sig2;                           # n⨯1              
%              = sum(_E_Xhat) .* -1/2 * sum(X-mu) ./ (sig2+eps).^(3/2);           
    _E_sig2    = -1/2 * sum(_E_Xhat.*(X-mu)) ./ (sig2+eps).^(3/2);     # n⨯1 
#   _sig2_mu   = -2/m*sum(X-mu) = -2/m*(sum(X)-m*mu) = -2*(mu-mu) = 0; # n⨯1    
#   _E_mu      = sum(_E_Xhat.*_Xhat_mu) .+ _E_sig2.*_sig2_mu;          # n⨯1               
#              = sum(_E_Xhat).*_Xhat_mu;                               # n⨯1               
    _E_mu      = -1 * sum(_E_Xhat)./sqrt(sig2+eps);                    # n⨯1               
    _E_X       = _E_Xhat.*_Xhat_X .+ _E_mu.*_mu_X .+ _E_sig2.*_sig2_X; # n⨯m
    dE         = _E_X;                                                 # n⨯m                        
    d.gam      = _E_gam / m;                                           # n⨯1             
    d.bet      = _E_bet / m;                                           # n⨯1             
    gg         = batchnorm_export(d);                                  # 1⨯2n                          
end

function ctx = batchnorm_update(ctx, gg)
    d        = ctx;
    d        = batchnorm_import(d, gg);
    ctx.gam -= d.gam;
    ctx.bet -= d.bet;
end

function pp = batchnorm_export(ctx)
    pp = [ ctx.gam(:); ctx.bet(:) ];
end

function ctx = batchnorm_import(ctx, pp)
    scope(ctx);
    n       = numel(gam);
    ctx.gam = reshape(pp(1:n),     size(gam));
    ctx.bet = reshape(pp(n+1:end), size(bet));
end
