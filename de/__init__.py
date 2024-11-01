import os
import sys
sys.path.append('../')

import torch
from torch import nn
from torch.nn import functional as F
from .model import Model

"""DE"""
class DE(object):
    """__init__"""
    def __init__(self, in_channels:int, n_class:int, m_sample:int, lr:float=1e-3, train_mode:bool=True, save_path="param", model_path="de_parameter.path"):
        self.in_channels = in_channels # チャンネル数
        self.n_class = n_class # クラス数
        self.m_sample = m_sample # サンプル数

        self.save_path = save_path # パラメータ保存先
        self.model_path = model_path # パラメータファイル名
        self.path = os.path.join(self.save_path, self.model_path) # パス生成
        
        self.mode = train_mode # 学習モード
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ディバイス設定
        self.model = Model(self.in_channels, self.n_class, self.m_sample).to(device=self.device) # Model

        if not os.path.exists(save_path):
            os.mkdir(self.save_path)

        if not self.mode:
            # パラメータファイルがない場合における処理を追加
            try:
                self.model.load_state_dict(torch.load(self.path))
            except:
                raise("Not Found model paramter file!!")
            

        self.lr = lr # 学習係数
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) # Optimizer
        self.losses = [] # 損失関数

    """tensor"""
    def tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    """numpy"""
    def numpy(self, x):
        if self.device == "cpu":
            return x.detach().numpy()
        else:
            return x.detach().cpu().numpy()
    
    """calc_parameters"""
    def calc_parameters(self, y, N):
        mu = torch.mean( y, dim=1 )
        var = torch.mean( torch.stack([ y[n].T @ y[n] - mu.T @ mu for n in range(N) ], dim=0), dim=1 )
        return mu, var

    """KLDLoss"""
    def KLD(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss

    """learn"""
    def learn(self, train_loader, epoch:int=5):
        self.model.train() # 学習モード
        for e in range(1, epoch+1):
            total_loss = 0.0
            for i, ( batch_data, batch_label ) in enumerate(train_loader):
                self.optim.zero_grad()
                N = batch_data.size(0)
                x, y = batch_data.to(device=self.device), F.one_hot(batch_label, self.n_class).float().to(device=self.device)
                y_pred = self.model(x)
                mu, var = self.calc_parameters(y_pred, N)
                loss = self.KLD((y - mu), torch.log( var ** 2) )
                loss.backward()
                total_loss += loss.item()
                self.optim.step()
            total_loss /= i
            self.losses.append(total_loss)
            print(f"epoch:{e}, loss:{total_loss}")

        if self.mode:
            torch.save(self.model.state_dict(), self.path)

    """predict"""
    def predict(self, x):
        self.model.eval() # 推論モード
        N = x.size(0)
        x = x.to(device = self.device)
        with torch.no_grad():
            y_pred = self.model(x)
            mu, var = self.calc_parameters(y_pred, N)
        return mu