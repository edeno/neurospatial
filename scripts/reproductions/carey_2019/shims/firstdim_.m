function d = firstdim_(x)
% First non-singleton dimension, as MATLAB's default reduction dimension.
d = find(size(x) ~= 1, 1); if isempty(d), d = 1; end
end
