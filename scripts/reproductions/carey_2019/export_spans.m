% Save each side's firstSpike/lastSpike of decS (the MakeQfromS range ends) for every session.
V = getenv('CAREY_VDM_CODE');  % folder holding the clones papers/ (3aebc8e) and vandermeerlab/ (ad0bbd4)
addpath(genpath(fullfile(V,'vandermeerlab','code-matlab','shared'))); warning('off','all');
d = dir(fullfile(root, 'sessions')); d = d([d.isdir] & ~startsWith({d.name}, '.'));
for i = 1:numel(d)
  f = fullfile(root, 'sessions', d(i).name, 'files');
  load(fullfile(f, 'octave_out.mat'));
  r = struct();
  for iC = 1:2
    r.(['first_' out.expCond(iC).label]) = firstSpike(out.expCond(iC).decS);
    r.(['last_' out.expCond(iC).label]) = lastSpike(out.expCond(iC).decS);
  end
  save('-v7', fullfile(f, 'spans.mat'), '-struct', 'r');
end
disp('ok')
