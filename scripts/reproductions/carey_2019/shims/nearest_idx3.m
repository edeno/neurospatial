function out = nearest_idx3(values, lookupvalues, direction)
% Octave port of vandermeerlab nearest_idx3.c (MvdM 2014): index into sorted
% lookupvalues of each value; direction -1 previous, 1 next, 0 nearest
% (a tie takes the later index); values outside the range clamp to 1 or n.
if nargin < 3, direction = 0; end
values = values(:); lookupvalues = lookupvalues(:);
n = numel(lookupvalues);
if n == 0, out = zeros(0, 1); return; end
out = zeros(numel(values), 1);
inside = values > lookupvalues(1) & values < lookupvalues(n);
k = lookup(lookupvalues, values(inside));        % table(k) <= v < table(k+1), 1-based
v = values(inside);
switch direction
  case 1
    o = k + 1;
  case -1
    o = k;
  otherwise
    o = k + 1;
    o(abs(lookupvalues(k) - v) < abs(lookupvalues(k + 1) - v)) = k(abs(lookupvalues(k) - v) < abs(lookupvalues(k + 1) - v));
end
out(inside) = o;
out(values >= lookupvalues(n)) = n;
out(values <= lookupvalues(1)) = 1;
end
