# Q3 外部行为与事件信息表说明

所有 CSV 均使用 UTF-8 编码，按 `record + original_trial_index` 与
`src/C/q3/output/trial_table.csv` 对齐。试次编号从 0 开始。不要按行号
猜测对应关系。保留空值表示未知，不要填 0 代替未知。

## trial_truth.csv

列：`record`、`original_trial_index`、`correct_response_side`。
主分析依照已给规则，将通道8与通道9解码响应同号判为正确、异号判为错误；
该列为可选外部核验真值，填 `left/right` 或 `-1/+1`，并在 `truth_source`
注明来源。
`authoritative_omission` 可选，只有实验日志明确标注漏答时才填 true/false。

## event_log.csv（可选）

Q1/Q3 试次表已经有 cue+2.2 s 的目标排程锚点，脚本会自动沿用并标记为
“排程假设”，不需要重复填写。若有独立目标事件日志，才需要补充真实时刻。
推荐宽表：每试次一行，列为
`record, original_trial_index, target_onset_time_s, response_deadline_time_s, timing_source`。时间是 MAT `TimeStamp` 的同一
绝对秒时钟；截止时间填截止时刻的绝对时间，不填持续时长。截止时间可以
留空。也接受长表：`record, original_trial_index, event_type, timestamp_s`，
其中 event_type 使用 `target_onset` 或 `response_deadline`。

## record_mapping.csv（可选）

每份 MAT 一行：`record, participant_id, task_id, session_id, mapping_source`。
按实验登记表填写，不能根据文件名推断被试或真实任务编号。编码信息会另从
MAT 的 DataLabel、VisCue 事件取值和当前事件解析代码中审计。被试/session
映射只影响按被试或 session 分组解释，不影响逐试次左右正确性计算。

## 判定口径

脚本用 MAT 的 Action/TgtAct 响应事件作为实际选择，并按 MAT 通道标签中的
L/R 数值码解码。同号 cue/响应按既定规则判正确，异号判错误；Task-2 同一动作
段内的同号 ±1 起始状态和后续 ±2 声明码作为一次应答。及时应答定义为首次
通道9边沿不晚于 cue+3.0 s。没有响应标记时，只有记录完整覆盖 cue+3.0 s
才能标记为未及时；未覆盖时保留未知。真实 target onset 缺失时，cue-to-response
时长仍可计算，但传统 target-to-response RT 仍未知。
