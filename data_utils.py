import numpy as np
import scanpy as sc
from torch.utils.data import Dataset, DataLoader


def adata_process(adata, min_genes=200, min_cells=10, max_value=10, n_top_genes=6000):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=max_value)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    df = adata.to_df().clip(lower=0)
    adata.X = df
    return adata


def add_mask(adata, fraction):
    df = adata.to_df()
    idx = df.index
    for i in range(len(df)):
        non_zero_indices = df.iloc[i].index[df.iloc[i].values != 0]
        num_to_zero = int(len(non_zero_indices) * fraction)
        indices_to_zero = np.random.choice(non_zero_indices, size=num_to_zero, replace=False)
        df.loc[idx[i], indices_to_zero] = 0
    adata.X = df
    return adata


class CSDataSet(Dataset):
    def __init__(self, ctrl, stim):
        self.ctrl = ctrl.to_df().values
        self.stim = stim.to_df().values

    def __getitem__(self, index):
        x = self.ctrl[index, :]
        y = self.stim[index, :]
        return x, y

    def __len__(self):
        return self.ctrl.shape[0]


class AnnDataSet(Dataset):
    def __init__(self, adata):
        self.data = adata.to_df().values

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


def data_loader(adata, key_dic, bz=128, cell_to_pred='CD4T'):
    cell_type_key = key_dic['cell_type_key']
    condition_key = key_dic['condition_key']
    ctrl_key = key_dic['ctrl_key']
    stim_key = key_dic['stim_key']

    types = list(set(adata.obs[cell_type_key]))
    new_adata = adata[(adata.obs[cell_type_key] == 'none')]

    for cell_type in types:
        if cell_type != cell_to_pred:
            ctrl = adata[
                (adata.obs[cell_type_key] == cell_type)
                & (adata.obs[condition_key] == ctrl_key)
            ]
            stim = adata[
                (adata.obs[cell_type_key] == cell_type)
                & (adata.obs[condition_key] == stim_key)
            ]

            ctrl_ind = np.random.choice(
                range(ctrl.shape[0]), size=(int(ctrl.shape[0] / bz) + 1) * bz
            )
            stim_ind = np.random.choice(
                range(stim.shape[0]), size=(int(stim.shape[0] / bz) + 1) * bz
            )

            ctrl_adata = ctrl[ctrl_ind, :]
            stim_adata = stim[stim_ind, :]

            ctrl_adata.obs["label"] = [cell_type + "_ctrl"] * ctrl_adata.obs.shape[0]
            stim_adata.obs["label"] = [cell_type + "_stim"] * stim_adata.obs.shape[0]

            if new_adata.obs.shape[0] == 0:
                new_adata = ctrl_adata.concatenate(stim_adata)
            else:
                new_adata = new_adata.concatenate(ctrl_adata, stim_adata)
        else:
            ctrl = adata[
                (adata.obs[cell_type_key] == cell_type)
                & (adata.obs[condition_key] == ctrl_key)
            ]
            ctrl_ind = np.random.choice(
                range(ctrl.shape[0]), size=(int(ctrl.shape[0] / bz) + 1) * bz
            )
            ctrl_adata = ctrl[ctrl_ind, :]
            ctrl_adata.obs["label"] = [cell_type + "_ctrl"] * ctrl_adata.obs.shape[0]

            if new_adata.obs.shape[0] == 0:
                new_adata = ctrl_adata
            else:
                new_adata = new_adata.concatenate(ctrl_adata)

    train_set = AnnDataSet(new_adata)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bz,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader


def balancer(adata, type_key, max=True):
    class_names = np.unique(adata.obs[type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[type_key] == cls].shape[0]

    number = np.max(list(class_pop.values())) if max else np.min(list(class_pop.values()))
    index_all = []

    for cls in class_names:
        class_index = np.array(adata.obs[type_key] == cls)
        index_cls = np.nonzero(class_index)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), number)]
        index_all.append(index_cls_r)

    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data
