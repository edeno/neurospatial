function m = nanmean(x, dim)
% MATLAB statistics nanmean: mean ignoring NaN; an all-NaN slice is NaN.
if nargin < 2, dim = firstdim_(x); end
ok = ~isnan(x); x(~ok) = 0; m = sum(x, dim) ./ sum(ok, dim);
end
