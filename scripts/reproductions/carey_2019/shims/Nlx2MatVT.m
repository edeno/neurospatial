function [Timestamps, X, Y, Angles, Targets, Points, Header] = Nlx2MatVT(fn, varargin)
% Stand-in for the Neuralynx MEX: raw records exported by export_nlx.py.
d = load(fullfile(fileparts(fn), 'VT_export.mat'));
Timestamps = d.Timestamps; X = d.X; Y = d.Y; Angles = d.Angles;
Targets = d.Targets; Points = d.Points; Header = {};
end
