import warnings

import scanpy as sc

from evaluation import calculate_mae, evaluate_adata
from model import SCVAEAT

warnings.filterwarnings("ignore")


def main():
    adata = sc.read_h5ad(r"\data\*.h5ad")
    adata = sc.AnnData(adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    adata.obs_names_make_unique()

    print(adata)
    print("\ncondition counts:\n", adata.obs["condition"].value_counts())
    print("\ncell_type counts:\n", adata.obs["cell_type"].value_counts())
    key_dic = {
        "condition_key": "condition",
        "cell_type_key": "cell_type",
        "ctrl_key": "control",
        "stim_key": "stimulated",
        "pred_key": "predict",
    }
    cell_to_pred = "CD4T"

    model = SCVAEAT(input_dim=adata.n_vars, model_size=adata.n_vars, device="cuda:0")
    model = model.to(model.device)

    train = adata[
        ~(
            (adata.obs[key_dic["cell_type_key"]] == cell_to_pred)
            & (adata.obs[key_dic["condition_key"]] == key_dic["stim_key"])
        )
    ]
    model.train_scVAEAT(train, epochs=100)

    pred = model.predict(
        train_adata=train,
        cell_to_pred=cell_to_pred,
        key_dic=key_dic,
        ratio=0.005,
    )
    print(pred)

    ground_truth = adata[adata.obs[key_dic["cell_type_key"]] == cell_to_pred]
    eval_adata = ground_truth.concatenate(pred)

    evaluate_adata(
        eval_adata=eval_adata,
        cell_type=cell_to_pred,
        key_dic=key_dic,
    )

    mae_all, mae_top100 = calculate_mae(eval_adata, key_dic)
    print(f"MAE (all genes): {mae_all:.6f}")
    print(f"MAE (top 100 DEGs): {mae_top100:.6f}")


if __name__ == "__main__":
    main()
