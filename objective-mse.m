'MSE COST';

function ctx = mse_init()
    ctx.unit = 'mse';
end;

function E = mse_cost(ctx, Z, Y)
    m = count(Z);
    E = sum((Z - Y).^2) / 2 / m;
end

function dE = mse_gradient(ctx, Z, Y)
    dE = Z - Y;
end
