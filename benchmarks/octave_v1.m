% https://octave.org/doc/v4.2.0/Script-Files.html
% "Unlike a function file, a script file must not begin with the keyword function. If it does, Octave will assume that it is a function file, and that it defines a single function that should be evaluated as soon as it is defined."
% WTF is this!?

1;

function retval = to_log(i, res)
  eps = 1e-6;
  f = i / (res - 1) * (1 - 2 * eps) + eps;
  retval = log(f ./ (1 - f));
endfunction

function retval = to_normal(i)
  retval = 1 ./ (1 + exp(-i));
endfunction

pi2 = 2 * pi;
n_iter = 3;

for res = 2.^(6:14)
  x = single(meshgrid(0:(res-1)));
  y = x';
  
  im = zeros(res, res, 3, 'single');
  t0 = time();
  
  for _ = 1:n_iter
    subres = res / 4;
    i = to_log(bitand(x, subres - 1), subres);
    j = to_log(bitand(y, subres - 1), subres);
    
    d = exp(-0.35 * (i.^2 + j.^2));
    
    k1 = 1 + 2 * idivide(x, subres);
    k2 = 0.5 + 0.5 * idivide(y, subres);
    
    im(:,:,1) = to_normal(i + k1 .* d .* sin(j * k2 * pi2));
    im(:,:,2) = to_normal(j + k1 .* d .* cos(i * k2 * pi2));
    im(:,:,1) = to_normal((i + j) * 0.5);
    
    result = uint8(round(im * 255));
  endfor
  
  t1 = time();
  disp(sprintf('Done in %10.3f ms (%4d x %4d)', 1e3 * (t1 - t0) / n_iter, res, res));
endfor
