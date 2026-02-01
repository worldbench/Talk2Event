## Vis toolkit

### 1. 获取`*.pkl`结果
经过修改，运行`test.py`后会在`exps/record_results`目录下生成两个文件：
- `{modality}_{attribute}_results.json`: 保存指标
- `{modality}_{attribute}_results.pkl`: 保存推理结果


### 2. 可视化
可视化程序从`{modality}_{attribute}_results.pkl`中读取infos进行可视化

- 运行程序 `python vis_tools/active_window.py`
- 从左上方的下拉框中选择需要可视化的`*.pkl`文件，点击即可