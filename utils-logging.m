'LOGGING UTILS';

function str = sprintvar(var, val, fmt=[])
    if isempty(fmt)
        if isscalar(val)
            if isstruct(val)
            elseif isnumeric(val)
                if fix(val) == val
                    fmt = '%d';
                else
                    fmt = '%f';
                end
            end
        elseif ischar(val)
            fmt = '%s';
        elseif iscell(val)
        elseif ismatrix(val)
            fmt = '%s';
            val = mat2str(val);
        end
    end
    if isempty(fmt)
        fmt = '%s';
        val = disp(val);
    end
    fmt = sprintf('%s = %s', var, fmt);
    str = sprintf(fmt, val);
end

function str = sprintsize(var, val)
    sz  = size(val);
    str = sprintf('%s = %s %s', var, typeinfo(val), mat2str(sz));    
end

global logfile = [];
global lospos  = [];

function printlog(fmt, varargin)
    str = sprintf(fmt, varargin{:});
    printf(str);
    global logfile
    global logpos
    if is_valid_file_id(logfile)
        if fmt(1:2) == '\r'
            fseek(logfile, logpos - ftell(logfile), SEEK_CUR);
        end
        logpos = ftell(logfile);
        fprintf(logfile, str);
        fflush(logfile);
    end
end

function log2file(path)
    global logfile
    global logpos
    logfile = fopen(path, 'wt');
    lospos  = ftell(logfile);
end

function printvar(var, fmt=[], val=evalin('caller', var))
    printlog('%s\n', sprintvar(var, val, fmt))
end

function printsize(var, val=evalin('caller', var))
    printlog('%s\n', sprintsize(var, val))
end

function printstruct(var, val=evalin('caller', var))
    for [v,k] = val
        is_size = count(k) > 5 && k(1:5) == "size_";
        if isstruct(v)
            printstruct([var '.' k], v);
        elseif isscalar(v) || ischar(v) || is_size
            printvar([var '.' k], [], v);
        else 
            printsize([var '.' k], v);
        end
    end
end

function printunit(ctxs, unit, varargin)
    unit = assignunit(ctxs, unit);
    if exist(unit)
        n = count(varargin);
        for i = 1:n
            field = varargin{i};
            printvar(sprintf('%s.%s', unit, field));
        end
    end
end

function printunderscore(ctx)
    eval(sprintf('_ = ctx;'));
    printstruct('_');
end

function printmodel(var, CTX=evalin('caller', var))
    for i = 1:count(CTX.layers)
        printvar(sprintf('CTX.layers{%d}.unit', i)); 
        printunderscore(CTX.layers{i});
    end
    for i = 1:count(CTX.optimizers)
        printvar(sprintf('CTX.optimizers{%d}.unit', i)); 
        printunderscore(CTX.optimizers{i});
    end
    printvar('CTX.objective.unit'); 
    printunderscore(CTX.objective);
    printvar('CTX.num_p');
end

function path = tmp(ext)
    global ipynb;
    path = ['tmp/' ipynb '.' ext];
end

function [ok,msg] = pushnotify(msg)
   url = 'https://api.pushover.net/1/messages.json?token=ajhpgiiie25dhehjek63q5w2p36r1r&user=umobvvsyqwdhxtgfmued18q6qxfee8';
   [s,ok,msg] = urlread(url, "post", { 'message', msg});
end

function printall(CTX, X, Y)
    printsize('X'); printvar('X');
    printsize('Y'); printvar('Y');
    m              = count(X); printvar('m');
    [CTX, Z]       = forward(CTX, X); printvar('CTX.XXX'); printvar('Z');
    E              = cost(CTX, Z, Y); printvar('E');  
    opt_E          = optimize_cost(CTX, E, m); printvar('opt_E');
    dE             = gradient(CTX, Z, Y); printvar('dE');
    [CTX, ggg]     = backward(CTX, Z, dE); printvar('ggg');
    acc            = accuracy(CTX, Z, Y); printvar('acc');
    [CTX, opt_ggg] = optimize_gradient(CTX, ggg, m); printvar('opt_ggg');
end

function arrow = dir2arrow(dir)
    if dir > 0
        arrow = '↑';
    elseif dir < 0
        arrow = '↓';
    else
        arrow = ' ';
    end
end

function yesno = bool2yesno(x)
    if x
        yesno = 'yes';
    else
        yesno = 'no';
    end
end

global logtic = tic();
function showlog(interval, width, fmt, varargin)
    global logtic;
    if toc(logtic) > interval
        str = sprintf(fmt, varargin{:});
        str = padright(str, ' ', width);
        printlog(['\r' str]);
        logtic = tic();
    end
end

global starts_at;

function printstart()
    global starts_at;
    printlog('\n');
    starts_at = clock();
    printvar('datestr(starts_at)');
    printlog('\n');
end

function printend(notification=[])
    global ipynb starts_at;
    printlog('\n');
    ends_at = clock();
    printvar('datestr(ends_at)');
    duration = ends_at - starts_at;
    printvar('duration');
    if !isempty(notification)
        notifed = pushnotify(sprintf('%s: %s', ipynb, notification))
    end
    printlog('\n');    
end
