import csv
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader

from PSRP.problem_solvers.gnn.RL.model import IRPModel


class IRPAgent:
    def __init__(
        self,
        depot_dim: int = 2,
        node_dim: int = 5,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        num_attention_layers: int = 3,
        num_heads: int = 8,
        lr: float = 1e-4,
        seed: int = 69,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = IRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model = IRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save_model(self, episode: int, check_point_dir: str) -> None:

        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)

        if episode % 50 == 0 and episode != 0:
            torch.save(
                self.model.state_dict(), check_point_dir + f"model_epoch_{episode}.pt",
            )

    def step(self, env, rollouts: Tuple[bool, bool]):
        env_baseline = deepcopy(env)

        # Go through graph batch and get loss
        loss, log_prob, kpi = self.model(env, rollouts[0])

        with torch.no_grad():
            loss_b, _, _ = self.target_model(env_baseline, rollouts[0])

        return loss, loss_b, log_prob, kpi

    def evaluate(self, env):
        self.model.eval()

        with torch.no_grad():
            loss, _, kpi = self.model(env, rollout=True)

        return loss, kpi

    def baseline_update(self, env, batch_steps: int = 3):
        self.model.eval()
        self.target_model.eval()

        current_model_cost = []
        baseline_model_cost = []
        with torch.no_grad():
            for _ in range(batch_steps):
                loss, loss_b, _, _ = self.step(env, [False, True])

                current_model_cost.append(loss)
                baseline_model_cost.append(loss_b)

        current_model_cost = torch.cat(current_model_cost)
        baseline_model_cost = torch.cat(baseline_model_cost)
        advantage = ((current_model_cost - baseline_model_cost) * -1).mean()
        _, p_value = stats.ttest_rel(
            current_model_cost.tolist(), baseline_model_cost.tolist()
        )

        if advantage.item() <= 0 and p_value <= 0.05:
            print("replacing baceline")
            self.target_model.load_state_dict(self.model.state_dict())