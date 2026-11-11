% Run the authors' Carey 2019 decoded-sequence pipeline on one session.
% usage: octave-cli --eval "session_dir='...'; n_shuffles=[]; run_session"
V = getenv('CAREY_VDM_CODE');  % folder holding the clones papers/ (3aebc8e) and vandermeerlab/ (ad0bbd4)
restoredefaultpath;
addpath(genpath(fullfile(V,'vandermeerlab','code-matlab','shared')));
addpath(genpath(fullfile(V,'vandermeerlab','code-matlab','tasks','Alyssa_Tmaze')));
addpath(genpath(fullfile(V,'papers','Carey_etal_submitted')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'shims'));
warning('off','all');
cd(session_dir);
cfg_decSeq = [];
cfg_decSeq.minSeqLength = 10; cfg_decSeq.removeInterneurons = 0; cfg_decSeq.nMinNeurons = 4;
cfg_decSeq.nMaxNanSkipSequential = 0; cfg_decSeq.output_file_prefix = 'S1_';
cfg_decSeq.nShuffles = n_shuffles; cfg_decSeq.writeFiles = 0;
tic; out = Generate_DecSeqShuf(cfg_decSeq); fprintf('decode %.1f s\n', toc);
save('-v7', fullfile(session_dir, 'files', 'octave_out.mat'), 'out');
if isempty(n_shuffles), for iC = 1:2, out.expCond(iC).seq_iv_shuf = {}; end, end
cfg_cand = []; cfg_cand.suffix = ''; LoadCandidates(cfg_cand);
cfg = []; cfg.minActiveCells = 4; cfg.minlen = 0.05; cfg.RemoveOverlappingEvents = 1;
cfg.processShuffles = ~isempty(n_shuffles); cfg.shuffle_include_p = 0.05; cfg.evt = evt;
seq = getEligibleSequences(cfg, out);
r = struct();
for iC = 1:2
  e = out.expCond(iC);
  r.(['raw_' e.label]) = [e.seq_iv.tstart(:) e.seq_iv.tend(:)];
  r.(['map_t_' e.label]) = e.decode_map.tvec(:);
  r.(['map_z_' e.label]) = e.decode_map.data(:);
  r.(['tc_' e.label]) = e.tc.tc;
  r.(['units_' e.label]) = char(e.decS.label);
end
r.eligible = [seq.tstart(:) seq.tend(:) seq.usr.side(:)];
save('-v7', fullfile(session_dir, 'files', 'octave_result.mat'), '-struct', 'r');
fprintf('eligible left %d right %d\n', sum(seq.usr.side==1), sum(seq.usr.side==2));
