# -------------------------- 1. 环境依赖导入 --------------------------
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind  # 用于差异基因统计检验

# 设置全局参数（保证可视化结果一致）
sc.settings.verbosity = 3  # 显示详细运行日志
sc.settings.set_figure_params(dpi=100, facecolor='white')  # 图表分辨率与背景
plt.rcParams['font.sans-serif'] = ['Arial']  # 避免中文乱码


# -------------------------- 2. 数据加载与基础认知 --------------------------
# 读取赛题提供的 h5ad 数据（需替换为你的本地文件路径）
adata = sc.read_h5ad("train.h5ad")  # 文档指定数据集文件名为 train.h5ad

# 1. 查看数据基本结构（匹配文档中 adata 对象的关键属性）
print("="*50)
print("数据基本信息：")
print(f"细胞数量 × 基因数量：{adata.shape}")  # 应包含 14893 个细胞（文档指定）
print(f"\n细胞类型（cell_type）分布：")
print(adata.obs["cell_type"].value_counts())  # 文档指定 7 种主要免疫细胞
print(f"\n扰动条件（condition）分布：")
print(adata.obs["condition"].value_counts())  # 文档指定 'control'（扰动前）/'stimulated'（扰动后）
print(f"\n训练集/测试集（split）分布：")
print(adata.obs["split"].value_counts())  # 文档指定 'train'/'test' 划分
print("="*50)

# 2. 数据校验（文档提示 count_matrix 已做 CPM 标准化，无需重复处理）
print("\n数据校验：")
print(f"基因表达矩阵是否存在空值：{np.isnan(adata.X).any()}")  # 检查空值
print(f"扰动前（control）细胞的基因表达量范围：{adata[adata.obs['condition']=='control'].X.min():.2f} ~ {adata[adata.obs['condition']=='control'].X.max():.2f}")
print(f"扰动后（stimulated）细胞的基因表达量范围：{adata[adata.obs['condition']=='stimulated'].X.min():.2f} ~ {adata[adata.obs['condition']=='stimulated'].X.max():.2f}")


# -------------------------- 3. 聚类与可视化（观察扰动对细胞分布的影响） --------------------------
# 文档要求：聚类（K-means/层次聚类）+ 降维（PCA/UMAP/t-SNE）展示扰动影响
# 步骤1：数据预处理（为聚类做准备，仅使用原始基因矩阵）
sc.pp.filter_genes(adata, min_cells=3)  # 过滤掉在少于 3 个细胞中表达的基因（减少噪声）
sc.pp.normalize_total(adata, target_sum=1e4)  # 补充标准化（避免极端值影响聚类）
sc.pp.log1p(adata)  # 对数转换（使数据更接近正态分布，提升聚类效果）

# 步骤2：降维（先 PCA 降维减少计算量，再 UMAP 降至 2 维用于可视化）
sc.pp.pca(adata, n_comps=50)  # PCA 降维到 50 维（保留主要信息）
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)  # 构建近邻图（聚类基础）
sc.tl.umap(adata)  # UMAP 降维到 2 维（文档推荐降维方法）

# 步骤3：聚类（使用层次聚类，scanpy 内置方法）
sc.tl.leiden(adata, resolution=0.5)  # Leiden 聚类（效果优于 K-means，可调整 resolution 控制聚类数量）

# 步骤4：可视化（按扰动条件、细胞类型、聚类结果分别绘图）
# 图1：按扰动条件（condition）着色，观察扰动对细胞分布的影响
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color="condition", ax=axes[0], title="UMAP (by Perturbation Condition)", show=False)
# 图2：按细胞类型（cell_type）着色，观察不同细胞类型的分布
sc.pl.umap(adata, color="cell_type", ax=axes[1], title="UMAP (by Cell Type)", legend_loc='on data', show=False)
# 图3：按聚类结果（leiden）着色，观察聚类与细胞类型/扰动的关联
sc.pl.umap(adata, color="leiden", ax=axes[2], title="UMAP (by Clustering Result)", show=False)
plt.tight_layout()
plt.savefig("perturbation_umap_visualization.png", dpi=300, bbox_inches='tight')  # 保存图片
plt.close()
print("\n聚类与可视化完成！图片已保存为 'perturbation_umap_visualization.png'")


# -------------------------- 4. 统计特征分析（识别扰动后显著变化的基因/细胞群体） --------------------------
# 文档要求：比较扰动前后的均值、方差，识别显著变化的基因
# 步骤1：按扰动条件分组，提取基因表达矩阵
control_cells = adata[adata.obs["condition"] == "control"]  # 扰动前细胞
stimulated_cells = adata[adata.obs["condition"] == "stimulated"]  # 扰动后细胞

# 步骤2：计算每组基因的统计特征（均值、方差）
# 扰动前基因统计
control_mean = control_cells.X.mean(axis=0).A1  # 每个基因的均值（A1 用于将矩阵转为一维数组）
control_var = control_cells.X.var(axis=0).A1    # 每个基因的方差
# 扰动后基因统计
stimulated_mean = stimulated_cells.X.mean(axis=0).A1
stimulated_var = stimulated_cells.X.var(axis=0).A1

# 步骤3：构建统计结果 DataFrame（便于后续分析）
gene_stats = pd.DataFrame({
    "gene_name": adata.var_names,  # 基因名
    "control_mean": control_mean,
    "control_var": control_var,
    "stimulated_mean": stimulated_mean,
    "stimulated_var": stimulated_var,
    "mean_diff": stimulated_mean - control_mean  # 扰动前后均值差异
})

# 步骤4：t检验识别显著差异基因（p<0.05 为显著）
# 对每个基因，比较扰动前后的表达量分布
p_values = []
for gene in adata.var_names:
    # 提取该基因在两组细胞中的表达量
    ctrl_expr = control_cells[:, gene].X.A1
    stim_expr = stimulated_cells[:, gene].X.A1
    # t检验（假设方差不齐）
    t_stat, p_val = ttest_ind(ctrl_expr, stim_expr, equal_var=False)
    p_values.append(p_val)

# 添加 p 值到统计结果，并标记显著差异基因
gene_stats["p_value"] = p_values
gene_stats["is_significant"] = gene_stats["p_value"] < 0.05  # p<0.05 标记为显著

# 步骤5：输出统计结果
print(f"\n统计特征分析结果：")
print(f"扰动前后显著差异基因数量（p<0.05）：{gene_stats['is_significant'].sum()}")
print(f"均值差异最大的前 5 个基因：")
print(gene_stats.nlargest(5, "mean_diff")[["gene_name", "control_mean", "stimulated_mean", "mean_diff", "p_value"]])
# 保存统计结果到 CSV（便于后续查看）
gene_stats.to_csv("gene_perturbation_stats.csv", index=False)
print("\n统计特征分析完成！结果已保存为 'gene_perturbation_stats.csv'")


# -------------------------- 5. 高变基因筛选（文档要求筛选前 1000 个高变基因） --------------------------
# 文档提示：可使用 sc.pp.highly_variable_genes 函数
# 步骤1：重新加载原始数据（避免之前的 log 转换影响高变基因计算）
adata_raw = sc.read_h5ad("train.h5ad")  # 重新读取原始数据
sc.pp.filter_genes(adata_raw, min_cells=3)  # 过滤低表达基因（与聚类步骤一致）

# 步骤2：筛选高变基因（n_top_genes=1000，文档明确要求）
sc.pp.highly_variable_genes(
    adata_raw,
    n_top_genes=1000,  # 前 1000 个高变基因
    flavor="seurat_v3",  # 采用 Seurat 算法（单细胞分析常用）
    layer="X",  # 使用原始表达矩阵（已做 CPM 标准化，文档提示）
    inplace=True  # 结果直接存入 adata_raw.var
)

# 步骤3：提取高变基因列表并验证
highly_var_genes = adata_raw.var[adata_raw.var["highly_variable"]].index.tolist()
print(f"\n高变基因筛选完成！共筛选出 {len(highly_var_genes)} 个高变基因（目标 1000 个）")
print(f"前 10 个高变基因：{highly_var_genes[:10]}")

# 保存高变基因列表到 CSV
pd.DataFrame({"highly_variable_gene": highly_var_genes}).to_csv("top1000_highly_variable_genes.csv", index=False)
print("高变基因列表已保存为 'top1000_highly_variable_genes.csv'")


# -------------------------- 6. 数据标准化与降维（构建建模用特征空间） --------------------------
# 文档要求：选择合适的标准化和降维方法，适配后续机器学习
# 步骤1：基于高变基因构建子矩阵（仅使用前 1000 个高变基因）
adata_model = adata_raw[:, highly_var_genes].copy()  # 仅保留高变基因

# 步骤2：Z-score 标准化（消除基因表达量级差异，文档推荐）
sc.pp.scale(adata_model, zero_center=True, max_value=10)  # zero_center=True 即 Z-score，max_value 限制极端值

# 步骤3：降维（PCA 降维，保留 50 维，平衡信息与效率）
sc.pp.pca(adata_model, n_comps=50, svd_solver="arpack")  # n_comps 可根据需求调整（50-100 均合适）

# 步骤4：验证降维结果（查看前 5 个主成分的解释方差比例）
pca_variance_ratio = adata_model.uns["pca"]["variance_ratio"]
print(f"\n降维完成！PCA 前 5 个主成分的解释方差比例：{pca_variance_ratio[:5]}")
print(f"PCA 前 50 个主成分累计解释方差比例：{np.sum(pca_variance_ratio):.4f}")  # 累计方差比例越高，信息保留越完整

# 步骤5：保存预处理后的建模数据（供任务二使用）
adata_model.write_h5ad("preprocessed_model_data.h5ad")  # 包含标准化后的高变基因矩阵 + PCA 结果
print("预处理后的建模数据已保存为 'preprocessed_model_data.h5ad'")


# -------------------------- 7. 流程结束提示 --------------------------
print("\n" + "="*60)
print("任务一：数据探索与预处理全流程完成！")
print("生成的关键文件清单：")
print("1. perturbation_umap_visualization.png → 聚类与扰动影响可视化图")
print("2. gene_perturbation_stats.csv → 基因扰动前后统计特征与差异分析结果")
print("3. top1000_highly_variable_genes.csv → 前 1000 个高变基因列表")
print("4. preprocessed_model_data.h5ad → 标准化+降维后的建模数据（供任务二使用）")
print("="*60)