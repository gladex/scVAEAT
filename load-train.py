# ### 1.Load data
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
adata = sc.read_h5ad('../data/sample.h5ad')
adata

adata = sc.AnnData(adata.X, obs=adata.obs.copy(), var=adata.var.copy())
adata.obs_names_make_unique()
print(adata)
print('\n', adata.obs['condition'].value_counts())
print('\n', adata.obs['cell_label'].value_counts())

# ### 2.Build and train models
import scanpy as sc
import numpy as np

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


# In[4]:


from torch.utils.data import Dataset, DataLoader

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
        x = self.data[index, :]
        return x

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
            ctrl = adata[((adata.obs[cell_type_key] == cell_type) &
                          (adata.obs[condition_key] == ctrl_key))]
            stim = adata[((adata.obs[cell_type_key] == cell_type) &
                          (adata.obs[condition_key] == stim_key))]

            ctrl_ind = np.random.choice(
                range(ctrl.shape[0]), size=(int(ctrl.shape[0] / bz) + 1) * bz)
            stim_ind = np.random.choice(
                range(stim.shape[0]), size=(int(stim.shape[0] / bz) + 1) * bz)
            ctrl_adata = ctrl[ctrl_ind, :]
            stim_adata = stim[stim_ind, :]
            ctrl_adata.obs["label"] = [
                cell_type + '_ctrl'] * ctrl_adata.obs.shape[0]
            stim_adata.obs["label"] = [
                cell_type + '_stim'] * stim_adata.obs.shape[0]
            if new_adata.obs.shape[0] == 0:
                new_adata = ctrl_adata.concatenate(stim_adata)
            else:
                new_adata = new_adata.concatenate(ctrl_adata, stim_adata)
        else:
            ctrl = adata[((adata.obs[cell_type_key] == cell_type) &
                          (adata.obs[condition_key] == ctrl_key))]
            ctrl_ind = np.random.choice(
                range(ctrl.shape[0]), size=(int(ctrl.shape[0] / bz) + 1) * bz)
            ctrl_adata = ctrl[ctrl_ind, :]
            ctrl_adata.obs["label"] = [
                cell_type + '_ctrl'] * ctrl_adata.obs.shape[0]
            if new_adata.obs.shape[0] == 0:
                new_adata = ctrl_adata
            else:
                new_adata = new_adata.concatenate(ctrl_adata)
    train_set = AnnDataSet(new_adata)
    train_loader = DataLoader(dataset=train_set, batch_size=bz, shuffle=False,
                              num_workers=0, drop_last=False)
    return train_loader


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import wasserstein_distance
from matplotlib.backends.backend_pdf import PdfPages

def evaluate_adata(eval_adata, cell_type, key_dic):
    sc.tl.pca(eval_adata)
    sc.pl.pca(eval_adata, color=key_dic['condition_key'], frameon=False,
              title="PCA of " + cell_type + " by Condition")
    sc.tl.rank_genes_groups(eval_adata, groupby=key_dic['condition_key'],
                            reference=key_dic['ctrl_key'], method="wilcoxon")
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"][key_dic['pred_key']]
    degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][key_dic['stim_key']]
    common_degs = list(set(degs_ctrl[0:100]) & set(degs_pred[0:100]))
    common_nums = len(common_degs)
    print("common DEGs: ", common_nums)
    draw_reg_plot(eval_adata=eval_adata,
                  cell_type=cell_type,
                  reg_type='mean',
                  axis_keys={"x": "predict", "y": key_dic['stim_key']},
                  condition_key=key_dic['condition_key'],
                  gene_draw=degs_ctrl[:10],
                  top_gene_list=degs_ctrl[:100],
                  save_path=None,
                  title=None,
                  show=True,
                  return_fig=False,
                  fontsize=12
                  )
    gene = degs_ctrl[0]
    sc.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    gene = degs_ctrl[1]
    sc.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    gene = degs_ctrl[2]
    sc.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    sc.tl.rank_genes_groups(eval_adata, groupby=key_dic['condition_key'],
                            reference=key_dic['ctrl_key'], method="wilcoxon")
    sc.pl.rank_genes_groups(eval_adata, n_genes=25, sharey=False, show=True)

    marker_genes = degs_ctrl[0:20]
    sc.pl.dotplot(eval_adata, marker_genes, groupby=key_dic['condition_key'], show=True)


def get_pearson2(eval_adata, key_dic, n_degs=100, sample_ratio=0.8, times=100):

    stim_key = key_dic['stim_key']
    pred_key = key_dic['pred_key']
    ctrl_key = key_dic['ctrl_key']
    condition_key = key_dic['condition_key']
    sc.tl.rank_genes_groups(eval_adata, groupby=condition_key, reference=ctrl_key, method="wilcoxon")
    degs = eval_adata.uns["rank_genes_groups"]["names"][stim_key][:n_degs]
    df_stim = eval_adata[(eval_adata.obs[condition_key] == stim_key)].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key] == pred_key)].to_df()
    data = np.zeros((times, 4))
    for i in range(times):
        stim = df_stim.sample(frac=sample_ratio, random_state=i)
        pred = df_pred.sample(frac=sample_ratio, random_state=i)
        stim_mean = stim.mean().values.reshape(1, -1)
        pred_mean = pred.mean().values.reshape(1, -1)
        stim_var = stim.var().values.reshape(1, -1)
        pred_var = pred.var().values.reshape(1, -1)
        r2_mean = (np.corrcoef(stim_mean, pred_mean)[0, 1]) ** 2
        r2_var = (np.corrcoef(stim_var, pred_var)[0, 1]) ** 2
        stim_degs_mean = stim.loc[:, degs].mean().values.reshape(1, -1)
        pred_degs_mean = pred.loc[:, degs].mean().values.reshape(1, -1)
        stim_degs_var = stim.loc[:, degs].var().values.reshape(1, -1)
        pred_degs_var = pred.loc[:, degs].var().values.reshape(1, -1)
        r2_degs_mean = (np.corrcoef(stim_degs_mean, pred_degs_mean)[0, 1]) ** 2
        r2_degs_var = (np.corrcoef(stim_degs_var, pred_degs_var)[0, 1]) ** 2
        data[i, :] = [r2_mean, r2_var, r2_degs_mean, r2_degs_var]
    df = pd.DataFrame(data, columns=['r2_all_mean', 'r2_all_var', 'r2_degs_mean', 'r2_degs_var'])
    r2_mean = df.mean(axis=0)
    r2_std = df.std(axis=0)
    return r2_mean, r2_std

def draw_reg_plot(eval_adata,
                  cell_type,
                  reg_type='mean',
                  axis_keys={"x": "predict", "y": "stimulated"},
                  condition_key='condition',
                  gene_draw=None,
                  top_gene_list=None,
                  save_path=None,
                  title=None,
                  show=True,
                  return_fig=False,
                  fontsize=14):

    df_case = eval_adata[(eval_adata.obs[condition_key]== axis_keys["y"])].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key]== axis_keys["x"])].to_df()

    if reg_type == 'mean':
        mean_case = df_case.mean().values.reshape(-1, 1)
        mean_pred = df_pred.mean().values.reshape(-1, 1)
    elif reg_type == 'var':
        mean_case = df_case.var().values.reshape(-1, 1)
        mean_pred = df_pred.var().values.reshape(-1, 1)
    data = np.hstack((mean_case, mean_pred))
    data_df = pd.DataFrame(data, columns=['case', 'predict'], index=df_case.columns)

    fig, ax = plt.subplots()
    sns.set_theme(color_codes=True)
    sns.regplot(x='case', y='predict', data=data_df, ax=ax)
    if gene_draw is not None:
        texts = []
        x = mean_case
        y = mean_pred
        for i in gene_draw:
            j = eval_adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
        adjust_text(
            texts,
            x=x,
            y=y,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
            force_points=(0.0, 0.0),
        )
    if top_gene_list is not None:
        data_deg = data_df.loc[top_gene_list, :]
        r_top = round(data_deg['case'].corr(data_deg['predict'], method='pearson'), 3)
        xt = 0.1 * np.max(data_df['case'])
        yt = 0.85 * np.max(data_df['predict'])
        ax.text(xt, yt, s='$R^2_{top 100 genes}$=' + str(round(r_top * r_top, 3)), fontsize=fontsize, color='black')
    r = round(data_df['case'].corr(data_df['predict'], method='pearson'), 3)
    xt = 0.1 * np.max(data_df['case'])
    yt = 0.75 * np.max(data_df['predict'])
    ax.text(xt, yt, s='$R^2_{all genes}$=' + str(round(r * r, 3)), fontsize=fontsize, color='black')
    if title:
        plt.title(title)
    else:
        plt.title('The Linear Regression of True and Predict Expression ' + reg_type + ' of ' + cell_type)
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
        plt.close()
    if show:
        plt.show()
        plt.close()
    if return_fig:
        return [round(r * r, 3), round(r_top * r_top, 3), fig]
    else:
        return [round(r * r, 3), round(r_top * r_top, 3)]


def get_wasserstein_distance(eval_adata, case_key='stimulated', pred_key='pred', 
                             top_genes=None, cal_type='sum'):

    dist_list = []
    dist_list_top = []
    pred = eval_adata[(eval_adata.obs["condition"] == pred_key)].to_df()
    case = eval_adata[(eval_adata.obs["condition"] == case_key)].to_df()

    for i in range(pred.shape[1]):
        gene_pred = pred.iloc[:, i].values
        gene_case = case.iloc[:, i].values
        dist = wasserstein_distance(gene_pred, gene_case)
        dist_list.append(dist)

    if top_genes is None:
        res = None
        if cal_type == 'mean':
            res = np.mean(dist_list)
        elif cal_type == 'sum':
            res = np.sum(dist_list)
        return res
    else:
        for gene in top_genes:
            gene_pred = pred.loc[:, gene].values
            gene_case = case.loc[:, gene].values
            dist = wasserstein_distance(gene_pred, gene_case)
            dist_list_top.append(dist)
        res = None
        if cal_type == 'mean':
            res = [np.mean(dist_list), np.mean(dist_list_top)]
        elif cal_type == 'sum':
            res = [np.sum(dist_list), np.sum(dist_list_top)]
        return res


# #### Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ATTN(nn.Module):
    def __init__(self, dff_size, model_size, drop=0.1):
        super(ATTN, self).__init__()
        self.fc1 = nn.Linear(model_size, dff_size)
        self.act = F.gelu
        self.fc2 = nn.Linear(dff_size, dff_size)
        self.attn = AttnModule2(dff_size)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fc3 = nn.Linear(dff_size, model_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        x = self.attn(x)
        x = x.squeeze(1)
        x = self.drop(x)
        x = self.fc3(x)
        return x

class SPA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SPA, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)  
        self.avg_pool2 = nn.AdaptiveAvgPool1d(2)  
        self.avg_pool4 = nn.AdaptiveAvgPool1d(4) 

        self.fc = nn.Sequential(
            nn.Linear(channel * (1 + 2 + 4), channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, l, c = x.shape  
        x = x.transpose(1, 2)  
        y1 = self.avg_pool1(x).reshape(b, -1)  
        y2 = self.avg_pool2(x).reshape(b, -1)  
        y3 = self.avg_pool4(x).reshape(b, -1)  
        y = torch.cat([y1, y2, y3], dim=1)  
        y = self.fc(y).unsqueeze(1)  

        return y 

class AttnModule2(nn.Module):
    def __init__(self, model_size, act_ratio=0.25, act_fn=F.gelu, gate_fn=torch.sigmoid):
        super(AttnModule2, self).__init__()
        reduce_channels = int(model_size * act_ratio)
        self.norm = nn.LayerNorm(model_size)
        self.global_reduce = nn.Linear(model_size, reduce_channels)
        self.local_reduce = nn.Linear(model_size, reduce_channels)
        self.act_fn = act_fn
        self.spatial_select = nn.Linear(2 * reduce_channels, 1)
        self.gate_fn = gate_fn
        self.spa = SPA(model_size) 

    def forward(self, x):
        ori_x = x 
        x = self.norm(x)
        x_global = x.mean(dim=1, keepdim=True)  
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.spa(x)  

        s_attn_input = torch.cat(
            [x_local, x_global.repeat(1, x.size(1), 1)], dim=-1)  
        s_attn = self.spatial_select(s_attn_input)
        s_attn = self.gate_fn(s_attn)  

        attn = c_attn * s_attn  
        return ori_x * attn


# In[ ]:


def balancer(adata, type_key, max=True):
    class_names = np.unique(adata.obs[type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[type_key] == cls].shape[0]
    if max:
        number = np.max(list(class_pop.values()))
    else:
        number = np.min(list(class_pop.values()))
    index_all = []
    for cls in class_names:
        class_index = np.array(adata.obs[type_key] == cls)
        index_cls = np.nonzero(class_index)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), number)]
        index_all.append(index_cls_r)

    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data


# #### scVAEAT

import torch
import torch.nn as nn
from torch import Tensor, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import scanpy as sc
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import ot

class SCVAEAT(nn.Module):
    def __init__(self, input_dim=7000, latent_dim=100, hidden_dim=1000,
                 noise_rate=0.1, kl_weight=5e-4, model_size=7000,dff_size=128,device=None):

        super(SCVAEAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_rate = noise_rate
        self.kl_weight = kl_weight
        self.Sl1_loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.attnencoder = ATTN(dff_size,model_size).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode2(self, x):
        h = self.encoder(x)
        return h
    
    def encodeattn(self, x):       
        h = self.attnencoder(x)
        return h

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        noise = torch.randn_like(x)
        a = torch.randn_like(x)
        za, mua, logvara = self.encode(a)
        x_noisy = x + noise * self.noise_rate
        z, mu, logvar = self.encode(x_noisy)
        z = z + za * 1e-1                  
        mu = mu + mua * 1e-1
        logvar = logvar + logvara * 1e-1
        bias = self.encodeattn(x_noisy)
        bias = bias[:, :100]
        z = z + bias * 1e-1
        x_hat = self.decode(z)
        std = torch.exp(logvar / 2)
        loss_kl = kl(
            Normal(mu, std),
            Normal(0, 1)
        ).sum(dim=1)
        loss_rec = ((x - x_hat) ** 2).sum(dim=1)

        return x_hat, loss_rec, loss_kl

    def get_latent_adata(self, adata):
        device = self.device
        x = Tensor(adata.to_df().values).to(device)
        latent_z = self.encode(x)[0].cpu().detach().numpy()
        latent_adata = sc.AnnData(X=latent_z, obs=adata.obs.copy())
        return latent_adata

    def get_loss(self, x):
        x_hat, loss_rec, loss_kl = self.forward(x)
        return x_hat, loss_rec, loss_kl

    def train_scVAEAT(self, train_adata, epochs=100, batch_size=128, lr=5e-4):

        device = self.device
        pbar = tqdm(range(epochs))
        anndataset = AnnDataSet(train_adata)
        train_loader = DataLoader(anndataset, batch_size=batch_size, shuffle=True, drop_last=False)
        SCPRAM_loss, loss_rec, loss_kl, loss_smooth = 0, 0, 0, 0
        optim_SCPRAM = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in pbar:
            pbar.set_description("Training Epoch {}".format(epoch))
            for idx, x in enumerate(train_loader):
                x = x.to(device)
                x_hat, loss_rec, loss_kl = self.get_loss(x)
                loss_smooth = torch.nn.functional.smooth_l1_loss(x, x_hat, reduction='none').sum(dim=1)
                SCPRAM_loss = (0.5 * loss_rec + 0.5 * (loss_kl *self.kl_weight) + 0.01 * loss_smooth).mean()
                optim_SCPRAM.zero_grad()
                SCPRAM_loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), 10)
                optim_SCPRAM.step()
            pbar.set_postfix(SCPRAM_loss=SCPRAM_loss.item(), recon_loss=loss_rec.mean().item(),kl_loss=loss_kl.mean().item(),
                             smooth_l1_loss=loss_smooth.mean().item())

    def ot_predict(self, adata_train, cell_to_pred, key_dic):

        ctrl_adata = adata_train[(
            adata_train.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        z_train = self.get_latent_adata(adata_train)
        z_ctrl_adata = z_train[(
            z_train.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        z_stim_adata = z_train[(
            z_train.obs[key_dic['condition_key']] == key_dic['stim_key'])]
        z_ctrl = z_ctrl_adata.to_df().values
        z_stim = z_stim_adata.to_df().values

        M = ot.dist(z_ctrl, z_stim, metric='euclidean')
        G = ot.emd(torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
                   torch.ones(z_stim.shape[0]) / z_stim.shape[0],
                   torch.tensor(M), numItermax=1000000)
        z_pred = torch.mm(G, torch.tensor(z_stim)).numpy() * z_ctrl.shape[0]

        pred_x = self.decode(Tensor(z_pred).to(
            self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(
            X=pred_x, obs=ctrl_adata.obs.copy(), var=ctrl_adata.var.copy())
        pred = pred_adata[(
            pred_adata.obs[key_dic['cell_type_key']]) == cell_to_pred]
        pred.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred

    def cross_cell_predict(self, train_adata, cell_to_pred, key_dic, n_top=None):

        ctrl_adata = train_adata[((train_adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                                  (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        train_z = self.get_latent_adata(train_adata)
        ctrl_z = train_z[(train_z.obs[key_dic['condition_key']]
                          == key_dic['ctrl_key'])]
        stim_z = train_z[(train_z.obs[key_dic['condition_key']]
                          == key_dic['stim_key'])]
        print(ctrl_z.shape, stim_z.shape)
        ctrl_z = balancer(ctrl_z, key_dic['cell_type_key'])
        stim_z = balancer(stim_z, key_dic['cell_type_key'])
        eq = min(ctrl_z.X.shape[0], stim_z.X.shape[0])
        cd_ind = np.random.choice(
            range(ctrl_z.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(
            range(stim_z.shape[0]), size=eq, replace=False)
        ctrl_z = ctrl_z[cd_ind, :]
        stim_z = stim_z[stim_ind, :]
        types = set(train_adata.obs[key_dic['cell_type_key']])
        ctrl_list, stim_list = [], []
        for cell_type in types:
            if (cell_type == cell_to_pred) or (cell_type == 'isolated'):
                continue
            ctrl_m = ctrl_z[(ctrl_z.obs[key_dic['cell_type_key']]
                             == cell_type)].to_df().values.mean(axis=0)
            stim_m = stim_z[(stim_z.obs[key_dic['cell_type_key']]
                             == cell_type)].to_df().values.mean(axis=0)
            if len(ctrl_list) > 0:
                ctrl_list = np.vstack((ctrl_list, ctrl_m))
                stim_list = np.vstack((stim_list, stim_m))
            else:
                ctrl_list = ctrl_m
                stim_list = stim_m

        delta_list = np.array(
            stim_list - ctrl_list).reshape(-1, self.latent_dim)
        test_z = self.get_latent_adata(ctrl_adata).to_df().values
        cos_sim = cosine_similarity(np.array(test_z).reshape(-1, self.latent_dim),
                                    np.array(ctrl_list).reshape(-1, self.latent_dim))
        if n_top is None:
            cos_sim = normalize(cos_sim, axis=1, norm='l1').reshape(
                test_z.shape[0], -1)
            delta_pred = np.matmul(cos_sim, delta_list)
        else:
            top_indices = np.argsort(cos_sim)[0][-n_top:]
            normalized_weights = cos_sim[0][top_indices] / \
                np.sum(cos_sim[0][top_indices])
            delta_pred = np.sum(normalized_weights[:, np.newaxis] *
                                np.array(delta_list).reshape(-1, self.latent_dim)[top_indices], axis=0)
        pred_z = test_z + delta_pred
        pred_x = self.decode(Tensor(pred_z).to(self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(X=pred_x, obs=ctrl_adata.obs.copy(), var=ctrl_adata.var.copy())
        pred_adata.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred_adata

    def predict(self, train_adata, cell_to_pred, key_dic, ratio=0.05):

        ctrl_to_pred = train_adata[((train_adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                                    (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        ctrl_adata = train_adata[(train_adata.obs[key_dic['cell_type_key']] != cell_to_pred) &
                                 (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        stim_adata = train_adata[(
            train_adata.obs[key_dic['condition_key']] == key_dic['stim_key'])]

        ctrl = self.get_latent_adata(ctrl_adata).to_df().values
        stim = self.get_latent_adata(stim_adata).to_df().values
        M = ot.dist(stim, ctrl, metric='euclidean')
        G = ot.emd(torch.ones(stim.shape[0]) / stim.shape[0],
                   torch.ones(ctrl.shape[0]) / ctrl.shape[0],
                   torch.tensor(M), numItermax=100000)
        match_idx = torch.max(G, 0)[1].numpy()
        stim_new = stim[match_idx]
        delta_list = stim_new - ctrl
        test_z = self.get_latent_adata(ctrl_to_pred).to_df().values
        cos_sim = cosine_similarity(np.array(test_z).reshape(-1, self.latent_dim),
                                    np.array(ctrl).reshape(-1, self.latent_dim))
        n_top = int(np.ceil(ctrl.shape[0] * ratio))
        top_indices = np.argsort(cos_sim)[0][-n_top:]
        normalized_weights = cos_sim[0][top_indices] / \
            np.sum(cos_sim[0][top_indices])
        delta_pred = np.sum(normalized_weights[:, np.newaxis] *
                            np.array(delta_list).reshape(-1, self.latent_dim)[top_indices], axis=0)
        pred_z = test_z + delta_pred
        pred_x = self.decode(Tensor(pred_z).to(
            self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(
            X=pred_x, obs=ctrl_to_pred.obs.copy(), var=ctrl_to_pred.var.copy())
        pred_adata.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred_adata


# In[ ]:


model = SCVAEAT(input_dim=adata.n_vars, device=device)  # 'cuda:0'
model = model.to(model.device)

key_dic = {'condition_key': 'condition',
           'cell_type_key': 'cell_label',
           'ctrl_key': 'Control',
           'stim_key': 'Hpoly.Day10',
           'pred_key': 'predict',
           }
cell_to_pred = 'Stem'  # Predict different cell types

train = adata[~((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
               (adata.obs[key_dic['condition_key']] == key_dic['stim_key']))]
model.train_scVAEAT(train, epochs=100)

