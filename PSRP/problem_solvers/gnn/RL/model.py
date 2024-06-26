from typing import Tuple
import numpy as np
import torch
from torch import nn
from PSRP.problem_solvers.gnn.RL.decoder import GraphDecoder
from PSRP.problem_solvers.gnn.RL.encoder import GraphDemandEncoder


class IRPModel(nn.Module):
    def __init__(
        self,
        depot_dim: int,
        node_dim: int,
        emb_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        num_heads: int,
        log_base: int,
        loss_coef:int,
        normalize_loss: bool,
    ):
        super().__init__()
        self.log_base = log_base
        self.normalize_loss = normalize_loss
        self.loss_coef = loss_coef
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.node_dim = node_dim

        self.encoder = GraphDemandEncoder(
            depot_input_dim=depot_dim,
            node_input_dim=node_dim,
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )
        self.decoder = GraphDecoder(
            emb_dim=emb_dim, num_heads=8, v_dim=emb_dim, k_dim=emb_dim, products_count=int((node_dim-2)/3)
        )

        self.model = lambda x, mask, rollout: self.decoder(
            x, mask, rollout=rollout
        )  # remove encoding and make it do it once


    def forward(self, env, rollout=False) -> Tuple[float, float]:
        done = False

        # split state in graph state and vehicle state
        local_features, global_features, mask, depots = env.get_state()

        acc_dist_loss = torch.zeros(size=(local_features.shape[0],), device=self.device)
        acc_log_prob = torch.zeros(size=(local_features.shape[0],), device=self.device)


        while not done:
            emb = self.encoder(
              x=local_features, depot_mask=depots.bool()
            )

            actions, load_percent, log_prob = self.decoder(
                node_embs=emb,
                mask=mask,
                global_features=global_features,
                rollout=rollout,
            )

            loss, kpis, done = env.step(actions.cpu().numpy(), None)#load_percent.detach().cpu().numpy()

            acc_log_prob += log_prob.squeeze().to(self.device)

            acc_dist_loss += torch.tensor(loss, dtype=torch.float, device=self.device) 

            local_features, global_features, mask, depots = env.get_state()

        
        dry_runs_loss = env.loss_dry_runs 

        if self.normalize_loss:
            normalize_coef = env.num_nodes * env.products_count * env.days_count
            dry_runs_loss = dry_runs_loss / normalize_coef

        dry_runs_loss = np.log(dry_runs_loss + 1) / np.log(self.log_base) * self.loss_coef
        dry_runs_loss = torch.tensor(dry_runs_loss, dtype=torch.float, device=self.device)
        acc_loss = acc_dist_loss - dry_runs_loss

        self.decoder.reset()

        return acc_loss, acc_log_prob, env.get_kpis()
