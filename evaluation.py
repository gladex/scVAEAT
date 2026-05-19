import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import numpy as np
from adjustText import adjust_text


def draw_reg_plot(
    eval_adata,
    cell_type,
    reg_type="mean",
    axis_keys={"x": "predict", "y": "stimulated"},
    condition_key="condition",
    gene_draw=None,
    top_gene_list=None,
    save_path=None,
    title=None,
    show=True,
    return_fig=False,
    fontsize=14,
):
    df_case = eval_adata[(eval_adata.obs[condition_key] == axis_keys["y"])].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key] == axis_keys["x"])].to_df()

    if reg_type == "mean":
        mean_case = df_case.mean().values.reshape(-1, 1)
        mean_pred = df_pred.mean().values.reshape(-1, 1)
    elif reg_type == "var":
        mean_case = df_case.var().values.reshape(-1, 1)
        mean_pred = df_pred.var().values.reshape(-1, 1)
    else:
        raise ValueError("reg_type 只能是 'mean' 或 'var'。")

    data = np.hstack((mean_case, mean_pred))
    data_df = pd.DataFrame(data, columns=["case", "predict"], index=df_case.columns)

    fig, ax = plt.subplots()
    sns.set_theme(color_codes=True)
    sns.regplot(x="case", y="predict", data=data_df, ax=ax)

    if gene_draw is not None:
        texts = []
        x = mean_case
        y = mean_pred
        for gene in gene_draw:
            j = eval_adata.var_names.tolist().index(gene)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, gene, fontsize=11, color="black"))
            ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
        adjust_text(
            texts,
            x=x,
            y=y,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
            force_points=(0.0, 0.0),
        )

    r_top = None
    if top_gene_list is not None:
        data_deg = data_df.loc[top_gene_list, :]
        r_top = round(data_deg["case"].corr(data_deg["predict"], method="pearson"), 3)
        xt = 0.1 * np.max(data_df["case"])
        yt = 0.85 * np.max(data_df["predict"])
        ax.text(
            xt,
            yt,
            s="$R^2_{top 100 genes}$=" + str(round(r_top * r_top, 3)),
            fontsize=fontsize,
            color="black",
        )

    r = round(data_df["case"].corr(data_df["predict"], method="pearson"), 3)
    xt = 0.1 * np.max(data_df["case"])
    yt = 0.75 * np.max(data_df["predict"])
    ax.text(
        xt,
        yt,
        s="$R^2_{all genes}$=" + str(round(r * r, 3)),
        fontsize=fontsize,
        color="black",
    )

    if title:
        plt.title(title)
    else:
        plt.title(f"The Linear Regression of True and Predict Expression {reg_type} of {cell_type}")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()

    if show:
        plt.show()
        plt.close()

    if return_fig:
        return [round(r * r, 3), round(r_top * r_top, 3) if r_top is not None else None, fig]
    return [round(r * r, 3), round(r_top * r_top, 3) if r_top is not None else None]


def evaluate_adata(eval_adata, cell_type, key_dic):
    sc.tl.pca(eval_adata)
    sc.pl.pca(
        eval_adata,
        color=key_dic["condition_key"],
        frameon=False,
        title="PCA of " + cell_type + " by Condition",
    )

    sc.tl.rank_genes_groups(
        eval_adata,
        groupby=key_dic["condition_key"],
        reference=key_dic["ctrl_key"],
        method="wilcoxon",
    )
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"][key_dic["pred_key"]]
    degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][key_dic["stim_key"]]
    common_degs = list(set(degs_ctrl[0:100]) & set(degs_pred[0:100]))
    common_nums = len(common_degs)
    print("common DEGs:", common_nums)

    draw_reg_plot(
        eval_adata=eval_adata,
        cell_type=cell_type,
        reg_type="mean",
        axis_keys={"x": "predict", "y": key_dic["stim_key"]},
        condition_key=key_dic["condition_key"],
        gene_draw=degs_ctrl[:10],
        top_gene_list=degs_ctrl[:100],
        save_path=None,
        title=None,
        show=True,
        return_fig=False,
        fontsize=12,
    )

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    for gene in [degs_ctrl[0], degs_ctrl[1], degs_ctrl[2]]:
        fig, ax = plt.subplots(facecolor="white")
        sc.pl.violin(
            eval_adata,
            keys=gene,
            groupby=key_dic["condition_key"],
            ax=ax,
            order=[key_dic["ctrl_key"], key_dic["pred_key"], key_dic["stim_key"]],
            palette=["#3E93BA", "#F1701A", "#4AB34A"],
            show=False,
        )
        ax.set_facecolor("white")
        ax.grid(axis="y", color="lightgray", linestyle="-", linewidth=0.8)
        ax.set_title(ax.get_title(), fontname="Times New Roman", fontsize=22)
        ax.set_xlabel(ax.get_xlabel(), fontname="Times New Roman", fontsize=22)
        ax.set_ylabel(ax.get_ylabel(), fontname="Times New Roman", fontsize=22)
        ax.tick_params(axis="both", labelsize=22)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")
        plt.show()

    sc.tl.rank_genes_groups(
        eval_adata,
        groupby=key_dic["condition_key"],
        reference=key_dic["ctrl_key"],
        method="wilcoxon",
    )
    sc.pl.rank_genes_groups(eval_adata, n_genes=25, sharey=False, show=True)

    marker_genes = degs_ctrl[0:20]
    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.sans-serif": ["Times New Roman"],
            "font.size": 16,
            "axes.unicode_minus": False,
        }
    ):
        sc.pl.dotplot(
            eval_adata,
            marker_genes,
            groupby=key_dic["condition_key"],
            title=f"Marker genes of {cell_type}",
            show=True,
        )


def calculate_mae(eval_adata, key_dic):
    sc.tl.rank_genes_groups(
        eval_adata,
        groupby=key_dic["condition_key"],
        reference=key_dic["ctrl_key"],
        method="wilcoxon",
    )
    degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][key_dic["stim_key"]]

    df_true = eval_adata[eval_adata.obs[key_dic["condition_key"]] == key_dic["stim_key"]].to_df()
    df_pred = eval_adata[eval_adata.obs[key_dic["condition_key"]] == key_dic["pred_key"]].to_df()

    true_mean_all = df_true.mean()
    pred_mean_all = df_pred.mean()
    mae_all = np.mean(np.abs(true_mean_all - pred_mean_all))

    top100 = list(degs_ctrl[:100])
    true_mean_top100 = df_true[top100].mean()
    pred_mean_top100 = df_pred[top100].mean()
    mae_top100 = np.mean(np.abs(true_mean_top100 - pred_mean_top100))

    return mae_all, mae_top100
