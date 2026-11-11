function [EVTimeStamps, EventIDs, TTLs, EVExtras, EventStrings, EVHeader] = Nlx2MatEV(fn, varargin)
% Stand-in for the Neuralynx MEX: raw records exported by export_nlx.py.
d = load(fullfile(fileparts(fn), 'EV_export.mat'));
EVTimeStamps = d.EVTimeStamps; EventStrings = cellstr(d.EventStrings);
EventIDs = []; TTLs = []; EVExtras = []; EVHeader = {};
end
