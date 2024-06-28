import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import torch

class GraphDecoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        num_heads: int = 8,
        v_dim: int = 128,
        k_dim: int = 128,
        products_count = 1,
    ):
        super().__init__()

        self._first_node = nn.Parameter(torch.rand(1, 1, emb_dim))
        self._last_node = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=3 * emb_dim,
            num_heads=num_heads,
            kdim=k_dim,
            vdim=v_dim,
            batch_first=True,
        )

        self._kp = nn.Linear(emb_dim, emb_dim, bias=False)

        # self._att_output = nn.Linear(emb_dim*3, emb_dim, bias=False)
        self._att_output_node = nn.Linear(emb_dim*3, emb_dim, bias=False)
        self._att_output_fuel = nn.Linear(emb_dim*3, 3, bias=False)

        # project in context of [graph_emb, ]
        self._context_proj = nn.Linear(emb_dim * 2 + 2, emb_dim * 3, bias=False)
        

        self.first_ = None
        self.last_ = None
        self.first_step = True
        self.num_heads = num_heads

    def forward(
        self,
        node_embs: torch.Tensor,
        mask: torch.Tensor = None,
        global_features: torch.Tensor = None,
        C: int = 10,
        rollout: bool = False,
    ):
        batch_size, _, emb_dim = node_embs.shape

        graph_emb = torch.mean(
            node_embs, axis=1, keepdims=True
        )  # shape (batch, 1, emb)

        if self.first_ is None:
            self.first_ = self._first_node.repeat(batch_size, 1, 1)
            self.last_ = self._last_node.repeat(batch_size, 1, 1)

        k = self._kp(node_embs)

        # Create context with first, last node and graph embedding.
        # Where last is the node from last decoding step.
        if global_features is None:
            context = torch.cat([graph_emb, self.first_, self.last_], -1)
        else:
            context = torch.cat([graph_emb, self.last_, *global_features], -1)

            context = self._context_proj(context)

        attn_mask = mask.repeat(self.num_heads, 1).unsqueeze(1)

        q, _ = self.attention(context, node_embs, node_embs, attn_mask=attn_mask)
    

        q_node = self._att_output_node(q)
        q_fuel = self._att_output_fuel(q)

        u_node = torch.tanh(q_node.bmm(k.transpose(-2, -1)) / emb_dim ** 0.5) * C
        u_fuel = torch.tanh(q_fuel / emb_dim ** 0.5) * C

        u_node = u_node.masked_fill(mask.unsqueeze(1).bool(), float("-inf"))

        log_prob = torch.zeros(size=(batch_size,))
        nn_idx = None
        f_idx = None

        if rollout:
            nn_idx = u_node.argmax(-1)
            f_idx = u_fuel.argmax(-1)
        else:
            m_node = Categorical(logits=u_node)
            m_fuel = Categorical(logits=u_fuel)
            nn_idx = m_node.sample()
            f_idx = m_fuel.sample()
            log_prob = (m_node.log_prob(nn_idx) + 
                    m_fuel.log_prob(f_idx))

        temp = nn_idx.unsqueeze(-1).repeat(1, 1, emb_dim)
        self.last_ = torch.gather(node_embs, 1, temp)

        if self.first_step:
            self.first_ = self.last_
            self.first_step = False

        return (nn_idx,f_idx), None, log_prob

    def reset(self):
        self.first_ = None
        self.last_ = None
        self.first_step = True