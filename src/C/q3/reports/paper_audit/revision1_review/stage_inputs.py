from pathlib import Path
import json, shutil, re
root = Path(r'D:\8\Desktop\CMathc\src\C\q3')
base = root / 'reports' / 'paper_audit' / 'revision1_review'
inp = base / 'input'
out = base / 'output'
inp.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
src_tex = Path(r'D:\8\Desktop\CMathc\data\第二十三届中国研究生数学建模竞赛+-+中文题目\中文题目\C题\问题三改版1.tex')
pdf = Path(r'D:\8\Desktop\CMathc\tmp\pdfs\q3_revision1_review\review.pdf')
problem = Path(r'D:\8\Desktop\CMathc\data\第二十三届中国研究生数学建模竞赛+-+中文题目\中文题目\C题\服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx')
shutil.copy2(src_tex, inp / 'manuscript.tex')
shutil.copy2(pdf, inp / 'main_preview.pdf')
shutil.copy2(problem, inp / 'official_problem.docx')
for name in ['decision_contract.json','quality_validation.json','result_object.json','claim_registry.json','figure_registry.json','table_registry.json','references.bib']:
    shutil.copy2(root / 'reports' / 'paper_audit' / 'input' / name, inp / name)
shutil.copy2(root / 'output' / 'dynamic_heldout_validation' / 'validation_summary.json', inp / 'validation_summary.json')
shutil.copy2(root / 'output' / 'dynamic_heldout_validation' / 'dynamic_cv_metrics.csv', inp / 'dynamic_cv_metrics.csv')
shutil.copy2(root / 'output' / 'continuation_audit' / 'event_timing_audit_summary.json', inp / 'event_timing_audit_summary.json')
shutil.copy2(root / 'output' / 'behavior_event_audit' / 'behavior_event_audit_summary.json', inp / 'behavior_event_audit_summary.json')
tex = src_tex.read_text(encoding='utf-8')
match = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', tex, re.S)
(inp / 'references_inline.txt').write_text(match.group(0) if match else 'No inline bibliography block found.', encoding='utf-8')
(inp / 'main_docx_not_applicable.json').write_text(json.dumps({'applicable': False, 'reason': 'Reviewed artifact is TeX; no DOCX version of this revision was supplied or generated.', 'rendered_artifact_used': 'main_preview.pdf'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(inp / 'project_manifest.json').write_text(json.dumps({'project_id': 'C-Q3-dynamic-paper', 'project_root': str(root), 'manuscript_scope': 'Problem 3 revision TeX; review limited to this question-level manuscript.', 'implementation': ['dynamic_cognitive_model.py', 'dynamic_validation.py', '02b_preprocess_raw.py', '03_extract_features.py'], 'evidence': ['output/dynamic_heldout_validation/validation_summary.json', 'output/continuation_audit/event_timing_audit_summary.json', 'output/behavior_event_audit/behavior_event_audit_summary.json']}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(inp / 'cumcm_layout_validation.json').write_text(json.dumps({'status': 'FAIL', 'pages': 11, 'source': 'XeLaTeX preview compiled from staged manuscript copy', 'findings': ['All six includegraphics render as black placeholders because graphicx is loaded with demo and image files are not resolved.', 'First-level headings are left-aligned.', 'Running section header appears at upper left.', 'Bibliography begins at the foot of page 10 and its final item is alone on page 11.', 'One overfull hbox warning around source line 210.']}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(inp / 'latex_compile_report.json').write_text(json.dumps({'status': 'PASS_WITH_WARNINGS', 'engine': 'XeLaTeX', 'passes': 2, 'page_count': 11, 'warnings': ['One overfull hbox at source line 210; inspect equation wrapping.'], 'visual_preview': 'Figures are placeholders due to [demo] graphicx option and unresolved fig1..fig6 paths.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(inp / 'docx_render_report.json').write_text(json.dumps({'status': 'NOT_APPLICABLE', 'reason': 'The reviewed revision is supplied as TeX; no DOCX artifact exists.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(inp / 'visual_layout_audit.json').write_text(json.dumps({'status': 'FAIL', 'preview': 'main_preview.pdf', 'pages_inspected': list(range(1, 12)), 'findings': ['Six black placeholder figures.', 'References split across pages 10 and 11.', 'Left-aligned first-level headings and upper-left running header.', 'Body text otherwise legible in the inspected preview.']}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
shutil.copy2(Path(r'D:\8\Desktop\CMathc\tmp\pdfs\q3_revision1_review\review.log'), inp / 'latex_compile.log')
print(base)