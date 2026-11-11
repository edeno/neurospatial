function [b, bint, r, rint, stats] = regress(y, X)
% Minimal MATLAB regress: least-squares b and stats = [R^2 F p errvar].
% A constant y gives R^2, F and p of NaN, as MATLAB does. bint, rint not computed.
y = y(:); b = X \ y; r = y - X * b; n = numel(y); p = size(X, 2);
sst = sum((y - mean(y)).^2); sse = sum(r.^2); r2 = 1 - sse / sst;
F = (r2 / (p - 1)) / ((1 - r2) / (n - p));
x = (n - p) / ((n - p) + (p - 1) * F);
if isfinite(x) && x >= 0 && x <= 1
  pval = betainc(x, (n - p) / 2, (p - 1) / 2);
else
  pval = NaN; if ~isfinite(r2), r2 = NaN; F = NaN; end
end
stats = [r2 F pval sse / (n - p)]; bint = []; rint = [];
end
