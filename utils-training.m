'TRAINING UTILS';

source('utils-logging.m');

function CHECKGRAD(CTX, BBB)
    [GRAD_DIFF, ~, params_checked] = check_gradient(CTX, pick(BBB));
    printvar('GRAD_DIFF');
    printvar('params_checked');
end

function [CTX, ok] = TRAIN(CTX, BBB, times=100, width=100)
    printlog('\n');

    starts_at = datestr(clock());
    printvar('starts_at');
    
    CHECKGRAD(CTX, BBB);
    
    dir   = nan;
    EEE   = zeros(1, times);
    ACC   = zeros(1, times);
    times = 1:times;
    for t = times
        [CTX, E, acc] = train(CTX, BBB);
        EEE(t)   = E;
        ACC(t)   = acc;
        if t > 1 
            sgn = sign(E - EEE(t-1));
        else
            sgn = 0;
        end
        if sgn != dir
            printlog('\n');
            dir = sgn;
        end
        msg = {sprintf('%d: %f %s', t, E, dir2arrow(dir))};
        if !isnan(acc)
            msg{end+1} = sprintf('accuracy: %f', acc);
        end    
        ctx = getunit(CTX.optimizers, 'gradient_clipping');
        if !isempty(ctx)
            msg{end+1} = sprintf('gradient-norm: %f', ctx.norm);
        end
        ctx = getunit(CTX.optimizers, 'stats');
        if !isempty(ctx)
            msg{end+1} = sprintf('update-ratio: %f', ctx.ratio);
        end
        msg = list_in_columns(msg, width);
        msg = msg(1:end-1);
        msg = padright(msg, ' ', width);
        printlog('\r%s', msg);
        if E == 0 || acc == 1.0
            break;
        end
    end
    printlog('\n\n');
    
    CHECKGRAD(CTX, BBB);
      
    figure('Position', [0 0 1000 400]);
    hold on;
    plot(1:t, EEE(1:t), 'r');
    plot(1:t, ACC(1:t), 'g');
    legend ('objective', "accuracy")
    title('training history');

    dirE = sign(EEE(t) - EEE(1));
    dirA = sign(ACC(t) - ACC(1));
    printlog('overall: objective %s, accuracy %s\n', dir2arrow(dirE), dir2arrow(dirA));

    ends_at = datestr(clock());
    printvar('ends_at');
    printlog('\n');
    
    ok = dirE < 0;
end

function CTX = TUNE(CTX, varargin)
    n = count(varargin);
    for i = 1:n
        args = varargin{i};
        CTX  = tune(CTX, args{:});
        printunit(CTX.layers,     args{1}, args{2});
        printunit(CTX.optimizers, args{1}, args{2});
    end
end
