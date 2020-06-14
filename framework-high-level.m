'HIGH-LEVEL FRAMEWORK';

function BBB = minibatches(X, Y, len=count(X))
    BBB = {};
    m   = count(X);
    n   = idivide(m, len, 'fix');
    for i = 1:n
        b      = i * len;
        a      = b - len + 1;
        B      = struct();
        B.X    = slice(X, [], a:b);
        B.Y    = slice(Y, [], a:b);
        BBB{i} = B;
    end
    rest = n * len + 1;
    if rest < m + 1
        B.X      = slice(X, [], rest:m);
        B.Y      = slice(Y, [], rest:m);
        BBB{n+1} = B;
    end
end

function [CTX, ggg, E, acc] = pretrain(CTX, X, Y)
    m          = count(X);
    [CTX, Z]   = forward(CTX, X);
    E          = cost(CTX, Z, Y);   
    E          = optimize_cost(CTX, E, m);
    dE         = gradient(CTX, Z, Y);
    [CTX, ggg] = backward(CTX, Z, dE);
    acc        = accuracy(CTX, Z, Y);
end

function [CTX, E, acc] = train(CTX, BBB)
    EE  = 0;
    ACC = 0;
    n   = length(BBB);
    for i = 1:n
        B                  = BBB{i};
        m                  = count(B.X);
        [CTX, ggg, E, acc] = pretrain(CTX, B.X, B.Y);
        [CTX, ggg]         = optimize_gradient(CTX, ggg, m);
        CTX                = update(CTX, ggg);
        EE                += E;
        ACC               += acc;
    end
    E   = EE  / n;
    acc = ACC / n;
end

function [diff, ddd, k] = check_gradient(CTX, B, timeout=5, eps=1e-5)
    ppp            = export(CTX);
    [CTX, ggg, E]  = pretrain(CTX, B.X, B.Y);
    [diff, ddd, k] = gradient_check(CTX, ggg, B.X, B.Y, timeout, eps);
end

function ms = measure_binary_classification(Z, Y, class=1)
    ms = struct();
    m  = ms.total = count(Z);
    TP = ms.TP = count(Z(:, Y == class & Y == Z));
    TN = ms.TN = count(Z(:, Y != class & Y == Z));
    FP = ms.FP = count(Z(:, Y == class & Y != Z));
    FN = ms.FN = count(Z(:, Y != class & Y != Z));
    if TP > 0
        pre = ms.precision = TP / (TP + FP); 
        rec = ms.recall    = TP / (TP + FN);
        ms.F1        = 2 / (1/pre + 1/rec);
        ms.accuracy  = (TP + TN) / m;
    end
end
