'TIC-TAC-TOE GAME'

function n = state2num(s)
    f = 3.^(0:8);
    n = sum(s(:) .* f') + 1;
end

function s = num2state(n)
    s = dec2base(n-1,3);
    s = flip(s);
    s = padright(s,'0',9);
    s = reshape(s,3,3) - '0';
end

function oh = game2oh18(s) 
    s = s(:);
    oh = onehots(s', [1,2])(:);
end

function oh = game2oh27(s) 
    s = s(:);
    oh = [ s == 0; s == 1; s == 2 ];
end

function p = other(p)
    p = mod(p,2)+1;
end

function p = player(s,a)
    if nargin == 1
        n = nnz(s);
        p = mod(n,2) + 1;
    elseif nargin == 2
        [y,x] = ind2sub([3,3], a);
        p     = s(y,x);
    end
end

function s = game(s,a)
    if nargin == 0
        s = zeros(3,3);   
    elseif nargin == 2 && any(actions(s) == a)
        [y,x]  = ind2sub([3,3], a);
        s(y,x) = player(s);
    end
end

function aaa = actions(s)
    [y,x] = find(s == 0);
    n     = length(y);
    aaa   = zeros(1,n);
    for i = 1:n
        aaa(i) = sub2ind([3,3], y(i), x(i));
    end
end

function yes = isover(s)
    yes = nnz(s) == 9;
end

function win = iswin(s,a)
    winner = -1;
    p      = player(s,a);
    [y,x]  = ind2sub([3,3], a);
    row    = all(s(y, 1:3) == p);
    col    = all(s(1:3, x) == p);
    diag1  = (x-y == 0) && all(s(1:4:9) == p); 
    diag2  = (x+y == 4) && all(s(3:2:7) == p); 
    win    = row || col || diag1 || diag2; 
end

function [winner,s] = play1(pi1, pi2=pi1)
    pi = {pi1, pi2};
    s  = game();
    while true
        p = player(s);
        a = pi{p}(s);
        s = game(s, a);
        if iswin(s,a)
            winner = p;
            break;
        end
        if isover(s)
            winner = 0;
            break;
        end    
    end
end

function [wins,draws] = play(n, pi1, pi2=pi1)
    printlog('playing %d times %s vs %s\n', n, func2str(pi1), func2str(pi2));
    wins  = zeros(1,2);
    draws = 0;
    for i = 1:n
        winner = play1(pi1, pi2);
        if winner > 0
            wins(winner) += 1;
        else
            draws += 1;
        end
    end
    printvar('wins');
    printvar('draws');
end

function a = randompolicy(s)
    aaa = actions(s);
    a   = pick(aaa);
end
