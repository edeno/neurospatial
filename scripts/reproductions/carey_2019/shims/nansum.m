function s = nansum(x, dim)
% MATLAB statistics nansum: sum ignoring NaN; an all-NaN slice sums to 0.
if nargin < 2, dim = firstdim_(x); end
x(isnan(x)) = 0; s = sum(x, dim);
end
