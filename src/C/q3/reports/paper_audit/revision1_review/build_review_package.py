from pathlib import Path
import hashlib, json
root = Path.cwd()
base = root / 'reports' / 'paper_audit' / 'revision1_review'
inp, out = base / 'input', base / 'output'
out.mkdir(parents=True, exist_ok=True)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
findings = [
  {'severity':'BLOCKER','title':'行为标签仍含占位符，且把未核实字段写成已生成','evidence':'manuscript.tex:28, 41, 65, 67, 115, 282-283；behavior_event_audit_summary.json 显示正确性、目标起点、截止时间和完整映射均未覆盖，correctness_labels_present=0；event_timing_audit_summary.json 中 verified reaction time/correctness/omission 均为 0。题目附录可确认第9通道编码左右点击，但事件审计未确认该沿就是精确动作时间。','impact':'文中“正确/错误/未及时标签已生成”“应答前100ms已截取”“未及时试次删失处理”无法由当前项目证据支持，且[待填]未完成。','repair':'删除占位符和已完成式表述；在取得经核验的试次目标映射、行为日志及官方截止时间前，明确将反应时间、正确性和漏答状态列为未知。'},
  {'severity':'MAJOR','title':'核心问题所要求的应答前终点验证没有对应的动态留出结果','evidence':'manuscript.tex:276-278 声称在 [t_cue,t_act-100ms] 重复 Q2 与 Q2+H 留出验证。当前 dynamic_cv_metrics.csv 的 evaluation_window 只有 candidate_target_600ms、cue_stage、late_stage，没有 response-marker-preceding-100ms 窗口；event_timing_audit_summary 将精确RT记为未核实。','impact':'固定晚期窗[0.8,2.8)s包含集中在约2.215s的第9通道沿，不能替代题目建议的应答前终点验证；正文结果结论目前不可复现。','repair':'在逐试次锚点语义核验后，实际运行并报告该终点的同折留出结果；否则把它改称“通道9标记前100ms探索特征”，删除已完成验证的结论，并说明问题三该项验证尚未完成。'},
  {'severity':'MAJOR','title':'论文方程与代码不一致，且控制态P只在图注出现','evidence':'manuscript.tex:159 的H递推缺少(1-rho_H)，并用u_Q2直接代替代码中的V；dynamic_cognitive_model.py 的 integrate_macro_states 实现了含(1-rho)的V/H/P更新。代码观测方程还含P调制项与控制加载，论文结果图注manuscript.tex:239比较Q2+H+P，但状态方程/参数表/主结果表未给出该模型的完整实现和结果。','impact':'读者无法按文中公式复现结果，也无法核对图中控制态模型的评价。','repair':'逐项照代码补全V/H/P递推、参数及传感器观测方程，并补报Q2+H+P结果；或删去P相关比较与结论。'},
  {'severity':'MAJOR','title':'目标时刻敏感性单元数写错','evidence':'manuscript.tex:250, 258-267 将3种预处理×3个候选时刻描述为六个单元，称2正、4负；表格实际为3×3=9格，dynamic validation_summary.json 记录 positive_preprocessing_onset_cells=2、total_preprocessing_onset_cells=9。','impact':'敏感性结论的分母和负向单元数不正确。','repair':'统一改为9格中2格改善、7格未改善；重新核对摘要、表后解读和图6。'},
  {'severity':'MAJOR','title':'六张图在源稿中被强制替换为黑色占位框','evidence':'manuscript.tex:4 加载 graphicx 的 demo 选项，图路径仅写 fig1…fig6；11页 XeLaTeX 预览中六幅图均显示占位框。','impact':'读者看不到验证流程、数值对比和敏感性图，图文证据链不成立。','repair':'移除 demo 选项，将六个图路径绑定到实际图文件，再重新编译检查。'},
  {'severity':'MODERATE','title':'摘要和敏感性描述需要跟验证口径完全一致','evidence':'主晚期窗三种预处理下Q2+H的数值与validation_summary.json一致：因果1.6474对1.6479（改善0.03%），无滤波恶化0.78%，零相位恶化1.31%；但响应前窗口结果没有相应动态验证指标。','impact':'现有“不存在稳定记忆增益”的主结论有数据支持；把它扩展到应答前窗口则超出证据。','repair':'保留固定晚期窗结论，清楚标成该窗口结论；待完成响应锚点验证后再扩展范围。'},
  {'severity':'MODERATE','title':'引文和版面尚未整理到提交状态','evidence':'正文未检出\cite引用；参考文献从第10页底部开始，第5条单独落在第11页；一级标题左对齐并显示左上页眉；编译日志有约第210行的overfull hbox。','impact':'外部方法来源与论断没有明确连接，末页和竞赛版式不够完整。','repair':'正文中为模型来源和评价方法加入对应引用；让参考文献另起页并复核竞赛模板版式和长公式换行。'}
]
report = '''# 问题三改版1审稿记录\n\n## 结论\n\n**目前不能判为完整满足问题三要求。** 稿件的主线可保留：复用问题二前向视觉机制、建立功能性记忆状态、按完整记录留出，并以“未观察到稳定记忆增益”作为谨慎结论，这些内容与当前代码的大体方向一致。但问题三明确提出以应答前约100 ms作为候选终点并用脑电信号验证；这部分在文稿中写成已完成，当前代码结果却没有对应窗口的Q2与Q2+H动态留出指标。行为标签、公式和图片也需要修正后再作为完成稿。\n\n## 必须先改\n\n'''
for i, f in enumerate(findings, 1):
    report += f"### {i}. [{f['severity']}] {f['title']}\n\n- **证据：** {f['evidence']}\n- **影响：** {f['impact']}\n- **最小修复：** {f['repair']}\n\n"
report += '''## 可以保留的结果\n\n- 400个提示事件中保留355个试次、按四份完整记录轮流留出，以及三种预处理的主晚期窗NRMSE，与项目验证摘要相符。\n- Q2前向映射最大绝对差为5.95×10^-8；该矩阵只用于正向投影，不应写成从三电极反演五个源。\n- 因果滤波下加入H仅改善0.03%，无滤波和零相位条件下反而变差；将结论限定为“未见跨处理口径稳定增益”是合适的。\n- 论文已说明任务映射、目标时刻、正确性、受试者独立性和源定位的限制。应把这些限制前后一致地落实到方法和结果表述。\n\n## 审查范围\n\n审查对象为问题三 TeX 单章和其 XeLaTeX 预览，依据官方C题、`src/C/q3` 当前实现、动态验证摘要及事件/行为审计。DOCX检查不适用，因为本次修订只有 TeX。该结论是问题三内容审查，不等于整篇竞赛论文的提交审查。\n'''
(out / 'paper_review_report.md').write_text(report, encoding='utf-8')
independent = {
  'review_type':'second_pass_content_review', 'verdict':'NOT_READY',
  'scope':'question 3 manuscript against official task and current Q3 implementation',
  'confirmed_strengths':['Primary late-window values match the current validation summary.','Cautious no-stable-memory-gain conclusion is supported for the fixed late window.','Source localization and participant-independence limitations are disclosed.'],
  'blocking_findings':['Behavior section contains [待填] and claims labels/window handling not present in verified audit outputs.','Response-marker-preceding window has no matching dynamic held-out comparison in dynamic_cv_metrics.csv.'],
  'major_findings':['H recurrence and observation design do not fully reproduce the implemented V/H/P model.','Three-by-three sensitivity table is described as six cells instead of nine.','All six figures are placeholders in the rendered preview.'],
  'checked_sources':['input/official_problem.docx','input/manuscript.tex','dynamic_cognitive_model.py','dynamic_validation.py','input/validation_summary.json','input/dynamic_cv_metrics.csv','input/event_timing_audit_summary.json','input/behavior_event_audit_summary.json']
}
(out / 'independent_content_review.json').write_text(json.dumps(independent, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
roles = ['manuscript.tex','main_preview.pdf','official_problem.docx','decision_contract.json','quality_validation.json','result_object.json','validation_summary.json','dynamic_cv_metrics.csv','event_timing_audit_summary.json','behavior_event_audit_summary.json']
hashes = {name: sha(inp / name) for name in roles}
(out / 'review_hash_binding.json').write_text(json.dumps({'status':'BOUND_TO_REVIEWED_COPIES','sha256':hashes,'note':'The source TeX and compiled preview were copied into this review package before review; hashes identify those staged copies.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
readiness = {
 'status':'NOT_READY', 'submission_readiness':False,
 'gates':[
  {'gate':'problem_3_pre_response_validation','status':'FAIL','reason':'No response-marker-preceding dynamic held-out metrics found.'},
  {'gate':'behavior_labels_and_omissions','status':'FAIL','reason':'Correctness/deadline labels are unavailable; draft has unresolved placeholders.'},
  {'gate':'equations_match_implementation','status':'FAIL','reason':'H recurrence is inconsistent and P comparison is not fully specified.'},
  {'gate':'figures_render','status':'FAIL','reason':'Six images render as black placeholders.'},
  {'gate':'sensitivity_counts','status':'FAIL','reason':'Nine table cells are narrated as six.'},
  {'gate':'references_and_layout','status':'NEEDS_REVISION','reason':'No in-text cites; bibliography split across pages; heading/header differs from contest layout.'}
 ],
 'minimum_next_step':'Correct unsupported behavior claims and placeholders, implement or remove the response-endpoint validation claim, align equations/results to code, fix 9-cell narrative, and embed actual figures before a final review.'
}
(out / 'submission_readiness.json').write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('review outputs written')