# 低轨卫星星座建模：Python 实现

## 当前进度

- [x] 问题一：单星覆盖、星下点轨迹、单轨连续可见与重叠率
- [x] 问题二：多轨道面星座优化
- [x] 问题三：星间链路、最短时延和容量
- [x] 问题四：碰撞风险、可靠度和冗余成本

## 运行环境

- Python 3.13
- NumPy、Pandas、Matplotlib、SciPy

## 运行问题一

```powershell
python .\问题1_求解.py
```

运行问题二：

```powershell
python .\问题2_求解.py
```

运行问题三：

```powershell
python .\问题3_求解.py
```

结果保存到 `results/`，图表以 SVG 和 300 dpi PNG 双格式保存到 `figures/`，并生成 `figures/图表面板.html`。

## 参数口径

主模型使用严格球面几何：地球半径 6371 km、轨道高度 550 km、天线半锥角 40.46°。题面标称覆盖半径 506 km 作为稳健性对照，不与严格几何结果混用。
