'TIC-TAC-TOE TACTICS GAME'

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

function a = makeaction(y, x, i)
    j = sub2ind([3,3], y, x);
    a = 10*j + i;
end

function [y,x,i] = splitaction(a)
    i     = mod(a,10);
    j     = fix(a/10);
    [y,x] = ind2sub([3,3], j);
end

function p = player(s)
    n = nnz(s(:,:,1:9));
    p = mod(n,2) + 1;
end

function win = iswin(s, y, x, i)
    s      = s(:,:,i);    
    p      = s(y,x);
    row    = all(s(y, 1:3) == p);
    col    = all(s(1:3, x) == p);
    diag1  = (x-y == 0) && all(s(1:4:9) == p); 
    diag2  = (x+y == 4) && all(s(3:2:7) == p); 
    win    = row || col || diag1 || diag2; 
end

function yes = isover(s, i)
    s   = s(:,:,i);
    yes = nnz(s) == numel(s);
end

function [s,winner] = game(s,a)
    winner = [];
    if nargin == 0
        s = zeros(3,3,10);   
    elseif nargin == 2 && any(actions(s) == a)
        p       = player(s);
        [y,x,i] = splitaction(a);
        assert(s(y,x,i) == 0);
        s(y,x,i) = p;
        if iswin(s,y,x,i)          # maybe inner win
            [y,x] = ind2sub([3,3],i);
            if s(y,x,10) == 0      # wasn't already won
                s(y,x,10) = p;     # sure inner win
                if iswin(s,y,x,10) # outer win
                    winner = p;
                elseif isover(s,10)
                    winner = 0;
                end
            end
        end
        if isover(s,1:9)
            winner = 0;
        end
    end
end

function aaa = actions(s)
    aaa = zeros(1,0);
    for i = 1:9
        for y = 1:3
            for x = 1:3
                if s(y,x,i) == 0
                    aaa(end+1) = makeaction(y,x,i);
                end
            end
        end
    end
end

function [M,m] = game2mat(s)
    M = zeros(9,9);
    for i = 1:9
        [y,x] = ind2sub([3,3], i);
        y1 = (y-1)*3 + 1; y2 = y1 + 2;
        x1 = (x-1)*3 + 1; x2 = x1 + 2;
        M(y1:y2,x1:x2) = s(:,:,i);
    end
    m = s(:,:,10);
end

function [winner,s] = play1(pi1, pi2=pi1)
    winner = [];
    pi     = {pi1, pi2};
    s      = game();
    do
        p          = player(s);
        a          = pi{p}(s);
        [s,winner] = game(s,a);
    until !isempty(winner);
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
        showlog(1, 80, 'game %d, winner %d, wins %s, draws %d', i, winner, mat2str(wins), draws);
    end
    printlog('\n');
end

function a = randompolicy(s)
    aaa = actions(s);
    a   = pick(aaa);
end
