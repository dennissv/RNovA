import math
import torch
from torch import nn

from itertools import combinations_with_replacement
from utils.BasicClass import ResidueOnlyPeptide, Candidate_Residue_AA

import triton
import triton.language as tl

@triton.jit
def typological_sort_floyd_warshall(ADJ,
                                    stride_batch,stride_row,stride_col,
                                    N: tl.constexpr,
                                    BLOCK_SIZE: tl.constexpr):
    start_batch = tl.program_id(0)
    start_batch = ADJ + start_batch.to(tl.int64) * stride_batch
    for scan_n in range(2,N):
        for start_row in range(0,N-scan_n):
            start_pos = start_batch + start_row * stride_row + start_row * stride_col
            adj_ptr = start_pos + scan_n * stride_col

            row_ptr = start_pos + tl.arange(0,BLOCK_SIZE) * stride_col
            col_ptr = adj_ptr + tl.arange(0,BLOCK_SIZE) * stride_row
            mask_row = tl.arange(0,BLOCK_SIZE)<scan_n
            mask_col = tl.arange(0,BLOCK_SIZE)<scan_n
            mask_self = tl.arange(0,BLOCK_SIZE)<=0

            row = tl.load(row_ptr, mask=mask_row)
            col = tl.load(col_ptr, mask=mask_col)
            a = row!=0
            b = col!=0
            have_path = a & b
            have_path = have_path | mask_self
            dist = row+col
            dist = tl.where(have_path, dist, 0)
            long_dist = tl.max(dist)
            tl.store(adj_ptr,long_dist)

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        div_term = 10000**(torch.arange(0, dim, 2, dtype=torch.float64)/dim)
        self.register_buffer('div_term', div_term, persistent=False)

    def forward(self, dist):
        pe_sin = torch.sin(dist.unsqueeze(dim=-1) / self.div_term)
        pe_cos = torch.cos(dist.unsqueeze(dim=-1) / self.div_term)
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype('cuda')
            return torch.concat([pe_sin, pe_cos],dim=-1).to(autocast_dtype)
        else:
            return torch.concat([pe_sin, pe_cos],dim=-1).float()

class PathEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        all_edge_mass = [0,float('inf')]
        candidate_ptm_aa = Candidate_Residue_AA()
        normal_aa = set([candidate_ptm_aa[ptm_aa] for ptm_aa in candidate_ptm_aa.keys() if ptm_aa.count('|')<1 and ptm_aa!='C'])
        for num in range(1,4):
            for i in combinations_with_replacement(normal_aa,num):
                all_edge_mass.append(ResidueOnlyPeptide([a.amino_acid_name for a in i]).total_residue_mass)
        all_edge_mass = torch.unique(torch.tensor(all_edge_mass).round(decimals=2))
        self.register_buffer('all_edge_mass', all_edge_mass, persistent=False)
        self.dist_embedding_forward = nn.Embedding(cfg.data.peptide_max_len*2, cfg.model.encoder.rel_size, padding_idx=0)
        self.dist_embedding_reverse = nn.Embedding(cfg.data.peptide_max_len*2, cfg.model.encoder.rel_size, padding_idx=0)

    def forward(self, node_mass):
        mass_difference_byrow = torch.triu(node_mass[:,None,:]-node_mass[:,:,None],diagonal=1).round(decimals=2)
        edge = torch.searchsorted(self.all_edge_mass, mass_difference_byrow)
        edge[self.all_edge_mass[edge]!=mass_difference_byrow] = 0
        path_distance = (edge>0).long()
        block_size = int(2 ** math.ceil(math.log2(path_distance.size(1))))
        typological_sort_floyd_warshall[(path_distance.size(0),)](path_distance,
                                                                  path_distance.stride(0),path_distance.stride(1),path_distance.stride(2),
                                                                  N=path_distance.size(1),
                                                                  BLOCK_SIZE=block_size)
        path_distance = self.dist_embedding_forward(path_distance) + self.dist_embedding_reverse(path_distance.transpose(1,2))
        return path_distance
