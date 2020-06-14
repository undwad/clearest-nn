'PLOTTING UTILS';

function plot1(xx, varargin)
    if !isempty(xx)
        plot(xx(1, :), xx(2, :), varargin{:}); 
    end
end

function plot_data(caption, X, Y, classes, CTX=[])
    disp(caption); 
    nc = count(classes);
    hold on;
    if isempty(CTX)
        args = { { 'ks', 'MarkerSize', 8, 'markerfacecolor', 'c' }
               , { 'ko', 'MarkerSize', 8, 'markerfacecolor', 'y' } 
               , { 'kd', 'MarkerSize', 8, 'markerfacecolor', 'm'  }
               };
        for i = 1:nc
            class = classes(i);
            xx    = X(:, find(Y == class));
            plot1(xx, args{i}{:});
        end
    else
        x1 = min(X(1, :));
        xN = max(X(1, :));
        y1 = min(X(2, :));
        yN = max(X(2, :));
        [x, y] = meshgrid(x1:0.01:xN, y1:0.01:yN);
        xy  = [x(:) y(:)]';            
        Z   = predict(CTX, xy);
        Z   = Z > 0.5;
        if nc > 2
            Z = onecolds(Z, classes, 0);
        end
        args = { { 'co', 'MarkerSize', 8, 'markerfacecolor', 'c' }
               , { 'yo', 'MarkerSize', 8, 'markerfacecolor', 'y' } 
               , { 'mo', 'MarkerSize', 8, 'markerfacecolor', 'm'  }
               };
        for i = 1:nc
            class = classes(i);
            xx    = xy(:, find(Z == class));
            plot1(xx, args{i}{:});
        end
        
        Z = predict(CTX, X);
        Z = Z > 0.5;
        if nc > 2
            Z = onecolds(Z, classes, 0);
        end
        args0 = { { 'rs', 'MarkerSize', 8, 'markerfacecolor', 'c' }
                , { 'ro', 'MarkerSize', 8, 'markerfacecolor', 'y' } 
                , { 'rd', 'MarkerSize', 8, 'markerfacecolor', 'm' }
                };
        args1 = { { 'ks', 'MarkerSize', 8, 'markerfacecolor', 'c' }
                , { 'ko', 'MarkerSize', 8, 'markerfacecolor', 'y' } 
                , { 'kd', 'MarkerSize', 8, 'markerfacecolor', 'm' }
                };
        tot_accuracy = 0;
        tot_F1       = 0;
        for i = 1:nc
            class = classes(i)
            xx    = X(:, find(Y == class & Y != Z));
            plot1(xx, args0{i}{:});
            xx    = X(:, find(Y == class & Y == Z));
            plot1(xx, args1{i}{:});
            
            metrics = measure_binary_classification(Z, Y, class)
            if metrics.TP > 0
                tot_accuracy = tot_accuracy + metrics.accuracy;
                tot_F1       = tot_F1 + metrics.F1;
            end
        end
        
        average_accuracy = tot_accuracy / nc
        average_F1       = tot_F1       / nc
        caption = sprintf('%s %f (%f)', caption, average_accuracy, average_F1);
    end
    
    xlabel('x');
    ylabel('y');
    title(caption);
    disp('');
end

function plot_both(scope, train_X, train_Y, test_X, test_Y, classes, CTX=[])
    figure('Position', [0 0 1000 400]);
    subplot(1, 2, 1);
    plot_data(sprintf('train data %s', scope), train_X, train_Y, classes, CTX);
    subplot(1, 2, 2);
    plot_data(sprintf('test data %s', scope), test_X,  test_Y, classes, CTX);
end

