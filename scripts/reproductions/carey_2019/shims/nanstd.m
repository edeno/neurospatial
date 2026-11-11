function s = nanstd(x, varargin)
% MATLAB statistics nanstd (flag 0, default dim): std ignoring NaN.
dim = firstdim_(x); if numel(varargin) >= 2, dim = varargin{2}; end
m = nanmean(x, dim); ok = ~isnan(x); d = x - m; d(~ok) = 0;
s = sqrt(sum(d.^2, dim) ./ (sum(ok, dim) - 1));
end
