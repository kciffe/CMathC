
说明：

- 8 个数学建模 skills 已经安装/校验成功
- 主要 Python 依赖也能正常导入
- 你现在本地的项目目录是：`D:\8\Desktop\CMathc\math-model-agent`

接下来建议这样走：

1. 以后做完整数学建模题，直接把题目、附件、数据放到一个单独项目文件夹里，然后对我说：

```text
使用 run-modeling-project 以 competition 档完成这个数学建模项目
```

2. 如果只是想先判断这个题该用什么模型，就说：

```text
使用 select-model 分析这个题适合什么模型
```

3. 如果你已经有 Excel/CSV 数据，想先处理数据，就说：

```text
使用 analyze-model-data 分析这些数据
```

4. 如果你已经有模型思路，想让我写代码求解，就说：

```text
使用 solve-model 建模并求解
```

5. 如果结果出来了，要画论文图，就说：

```text
使用 make-model-figures 生成论文图表
```

6. 如果要写论文，就说：

```text
使用 write-model-paper 根据结果生成数学建模论文
```

7. 最后提交前检查：

```text
使用 review-model-paper 审查论文
```

最推荐你现在做的是：把一个具体赛题的题面、附件数据放进一个新文件夹，然后让我从 `$run-modeling-project` 开始带你跑完整流程。这个项目不是普通脚本库，它更像一套“数学建模比赛流水线”。