from pathlib import Path
import json
root = Path.cwd()
report_dir = root / 'src/C/q3/reports'
md = next(p for p in report_dir.glob('*.md') if p.read_text(encoding='utf-8').startswith('# 5.3 '))
run_outputs = root / 'tmp/q3_paper_run/outputs'
required = ['paper_generation_report.json','claim_usage_report.json','cumcm_layout_validation.json','render_request.json']
assert md.stat().st_size > 6000
body = md.read_text(encoding='utf-8')
assert '0.521' in body and '0.460' in body and 'DDM' in body
assert 'TODO' not in body and 'TBD' not in body
for relative in ['../output/features_by_cue_side.png','../output/eeg_heldout_record_confusion.png','../output/state_proxy_scores.png','figures/F11_scope_sensitivity.png']:
    assert (md.parent / relative).resolve().is_file(), relative
for name in required:
    json.loads((run_outputs/name).read_text(encoding='utf-8'))
assert list(run_outputs.glob('*.tex'))
print('Markdown, figure links, TeX fragment, and paper trace outputs are present.')
