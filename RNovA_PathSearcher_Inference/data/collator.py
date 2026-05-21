import torch
import torch.nn.functional as F
from utils.BasicClass import Candidate_Residue_AA

class RnovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.candidate_ptm_aa = Candidate_Residue_AA()
        self.candidate_normal_aa = set([self.candidate_ptm_aa[ptm_aa] for ptm_aa in self.candidate_ptm_aa.keys() if ptm_aa.count('|')<1 and ptm_aa!='I']+[self.candidate_ptm_aa['C|UniMod:4']])

    def __call__(self, batch):
        nodes_count = [record['node_mass'].size(0) for record in batch]
        max_nodes_count = max(nodes_count)
        node_size_bucket = sorted(self.cfg.data.node_size_bucket)
        for bucket_boundary in node_size_bucket:
            if max_nodes_count <= bucket_boundary:
                max_nodes_count = bucket_boundary
                break

        node_mass = torch.stack([F.pad(record['node_mass'], (0,max_nodes_count-nodes_count[i])) for i, record in enumerate(batch)])
        node_intensity = torch.stack([F.pad(record['node_intensity'], (0,max_nodes_count-nodes_count[i])) for i, record in enumerate(batch)])
        node_class = torch.stack([F.pad(record['node_class'], (0,max_nodes_count-nodes_count[i])) for i, record in enumerate(batch)])
        peak_intensity_rank = torch.stack([F.pad(record['peak_intensity_rank'], (0,max_nodes_count-nodes_count[i])) for i, record in enumerate(batch)])
        peak_moverz = torch.stack([F.pad(record['peak_moverz'], (0,max_nodes_count-nodes_count[i])) for i, record in enumerate(batch)])
        charge = torch.LongTensor([record['charge'] for record in batch]).unsqueeze(dim=-1)

        node_mask = torch.stack([F.pad(torch.ones_like(record['node_mass'], dtype=bool), (0,max_nodes_count-nodes_count[i])) for i, record in enumerate(batch)])
        node_mask[:,0] = False

        title = [record['title'] for record in batch]
        return ({'node_mass':node_mass,
                 'node_intensity':node_intensity,
                 'charge':charge,
                 'peak_intensity_rank':peak_intensity_rank,
                 'peak_moverz':peak_moverz,
                 'node_class':node_class},
                 node_mask, title)
