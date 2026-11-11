function [C, order] = confusionmat(g1, g2, varargin)
% Minimal MATLAB confusionmat: counts of (known, predicted) label pairs; supports 'order'.
order = [];
for i = 1:2:numel(varargin), if strcmpi(varargin{i}, 'order'), order = varargin{i+1}(:); end, end
g1 = g1(:); g2 = g2(:); ok = ~isnan(g1) & ~isnan(g2);
if isempty(order), order = unique([g1(ok); g2(ok)]); end
[~, a] = ismember(g1(ok), order); [~, b] = ismember(g2(ok), order); keep = a > 0 & b > 0;
C = accumarray([a(keep) b(keep)], 1, [numel(order) numel(order)]);
end
