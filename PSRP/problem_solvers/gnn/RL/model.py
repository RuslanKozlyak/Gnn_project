from typing import Tuple
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
    ):
        super().__init__()
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
            emb_dim=emb_dim, num_heads=8, v_dim=emb_dim, k_dim=emb_dim, products_count=int((node_dim-2)/4)
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

            actions, log_prob = self.decoder(
                node_embs=emb,
                mask=mask,
                global_features=global_features,
                rollout=rollout,
            )

            loss, kpis, done = env.step(actions.cpu().numpy())

            acc_log_prob += log_prob.squeeze().to(self.device)
            acc_dist_loss += torch.tensor(loss, dtype=torch.float, device=self.device)

            local_features, global_features, mask, depots = env.get_state()

        normalization_coef = (env.num_nodes - env.num_depots) * env.days_count *  env.products_count
        dry_runs_loss = env.loss_dry_runs #/ normalization_coef

        dry_runs_loss = torch.tensor(dry_runs_loss, dtype=torch.float, device=self.device)

        acc_loss = acc_dist_loss - dry_runs_loss

        self.decoder.reset()

        return acc_loss, acc_log_prob, env.get_kpis()
