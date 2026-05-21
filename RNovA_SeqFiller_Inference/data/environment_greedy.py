import torch
import numpy as np
from utils.BasicClass import Candidate_Residue_AA, Residual_AA
from .knapsack_build import knapsack_build_vsize32, knapsack_build_vsize64, next_aa_mask_builder

from math import ceil

class Environment(object):
    def __init__(self, cfg, model, inference_dl, device, candidate_amino_acids):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.device = device

        # Input prepration.
        self.seq = torch.ones([self.cfg.train.batch_size,1],dtype=torch.long,device=self.device)
        self.seq_pos = torch.zeros_like(self.seq)
        self.seq_mass_forward = torch.zeros_like(self.seq,dtype=torch.float)
        self.iter_num = torch.ones_like(self.seq)

        # Reserve 2 times of max len position for ensure
        # there is enough space in cache tensor.
        # And fill in first seq token.
        self.candidate_ptm_aa = Candidate_Residue_AA()
        candidate_aa = []
        for aa in candidate_amino_acids:
            try:
                candidate_aa.append(self.candidate_ptm_aa[aa])
            except KeyError:
                residue, delta_mass = _parse_dynamic_candidate_aa(aa, self.candidate_ptm_aa)
                new_temp_aa = Residual_AA(residue,
                                          n_terminal_PTM='',
                                          c_terminal_PTM='',
                                          r_group_PTM=str(delta_mass),
                                          embedding_db_index=len(self.candidate_ptm_aa)+3,
                                          composition=None,
                                          mass=delta_mass+self.candidate_ptm_aa[residue].mass,
                                          full_name=aa)
                self.candidate_ptm_aa.add_residue(aa,new_temp_aa)
                candidate_aa.append(self.candidate_ptm_aa[aa])

        candidate_aa = sorted(candidate_aa)
        self.candidate_aa = torch.tensor([aa.embedding_db_index for aa in candidate_aa],device='cuda',dtype=torch.long).unsqueeze(0)
        self.basic_candidate_aa = torch.tensor([self.candidate_ptm_aa[aa.amino_acid_name].embedding_db_index if aa.amino_acid_name!='C' else self.candidate_ptm_aa['C|UniMod:4'].embedding_db_index for aa in candidate_aa],device='cuda',dtype=torch.long).unsqueeze(0)
        self.candidate_mass = torch.tensor([aa.mass for aa in candidate_aa],device='cuda',dtype=torch.float).unsqueeze(0)
        self.candidate_mass_cpu = np.array([aa.mass for aa in candidate_aa])
        self.candidate_aa_num = self.candidate_aa.size(1)
        self.result_seq_ntoc = torch.zeros([self.cfg.train.batch_size,self.cfg.data.peptide_max_len*20], device=self.device, dtype=torch.long)
        self.result_seq_score_ntoc = torch.zeros_like(self.result_seq_ntoc, dtype=torch.float)
        self.result_seq_cton = torch.zeros_like(self.result_seq_ntoc)
        self.result_seq_score_cton = torch.zeros_like(self.result_seq_score_ntoc)

        self.aa_resolution = 10000
        if len(self.candidate_mass_cpu) <= 32:
            self.knapsack_matrix = knapsack_build_vsize32(self.candidate_mass_cpu,4000,self.aa_resolution)
        elif 32 < len(self.candidate_mass_cpu) <= 64:
            self.knapsack_matrix = knapsack_build_vsize64(self.candidate_mass_cpu,4000,self.aa_resolution)

        self.k_cache_self = torch.zeros(self.cfg.model.decoder.num_layers,
                                        self.cfg.train.batch_size,
                                        ceil(self.cfg.data.peptide_max_len*self.cfg.data.max_iter*4),
                                        self.cfg.model.decoder.num_heads,
                                        self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads,
                                        dtype=torch.float16,
                                        device=device)
        self.v_cache_self = torch.zeros(self.cfg.model.decoder.num_layers,
                                        self.cfg.train.batch_size,
                                        ceil(self.cfg.data.peptide_max_len*self.cfg.data.max_iter*4),
                                        self.cfg.model.decoder.num_heads,
                                        self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads,
                                        dtype=torch.float16,
                                        device=device)

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        candidate_aa, decoder_step_input, title = self.exploration_initializing()
        decoder_step_input = self.next_aa_choice(candidate_aa, decoder_step_input)

        while (self.iter_num<=self.cfg.data.max_iter).any():
            candidate_aa = self.model(**decoder_step_input)
            decoder_step_input = self.next_aa_choice(candidate_aa, decoder_step_input)
        result, result_score = self.result_generator()
        return result, result_score, title

    def exploration_initializing(self):
        node_input, precursor_mass, title = next(self.inference_dl)
        self.precursor_mass = precursor_mass

        self.iter_num[:] = 1

        self.seq = self.seq[:len(title)]
        self.seq_pos = self.seq_pos[:len(title)]
        self.iter_num = self.iter_num[:len(title)]
        self.seq_mass_forward = self.seq_mass_forward[:len(title)]
        self.k_cache_self = self.k_cache_self[:,:len(title)]
        self.v_cache_self = self.v_cache_self[:,:len(title)]
        self.result_seq_ntoc = self.result_seq_ntoc[:len(title)]
        self.result_seq_score_ntoc = self.result_seq_score_ntoc[:len(title)]
        self.result_seq_cton = self.result_seq_cton[:len(title)]
        self.result_seq_score_cton = self.result_seq_score_cton[:len(title)]

        sequence_input = {
            'seq': self.seq,
            'seq_pos': self.seq_pos,
            'seq_iter': self.iter_num,
            'seq_mass_forward': self.seq_mass_forward,
            # Constant Value for a batch
            'candidate_aa': self.basic_candidate_aa,
            'candidate_aa_mass': self.candidate_mass
        }

        k_cache, v_cache = self.model.encoder_forward(**node_input)
        candidate_aa = self.model(sequence_input, 0, self.k_cache_self, self.v_cache_self, k_cache, v_cache)

        decoder_step_input = {
            'cache_seqlens': 0,
            'sequence_input': sequence_input,
            'k_cache_self': self.k_cache_self,
            'v_cache_self': self.v_cache_self,
            'k_cache_cross': k_cache,
            'v_cache_cross': v_cache
        }
        self.result_seq_ntoc.zero_()
        self.result_seq_score_ntoc.zero_()
        self.result_seq_cton.zero_()
        self.result_seq_score_cton.zero_()
        return candidate_aa, decoder_step_input, title

    @torch.compile
    def next_aa_choice(self, candidate_aa, decoder_step_input):
        ms1_threshold = (self.precursor_mass*5e-6*self.aa_resolution).round().long().cpu()
        remain_mass = self.precursor_mass-decoder_step_input['sequence_input']['seq_mass_forward']
        remain_mass = (remain_mass*self.aa_resolution).round().long().cpu()
        inference_mask = []
        for x, y in zip(remain_mass, ms1_threshold):
            if x+y>len(self.knapsack_matrix):
                inference_mask += [torch.ones(self.candidate_mass.size(1),dtype=bool)]
            else:
                inference_mask += [torch.from_numpy(next_aa_mask_builder(self.knapsack_matrix[x-y:x+y],len(self.candidate_mass_cpu)))]
        inference_mask = torch.stack(inference_mask).to(candidate_aa.device)
        candidate_aa = candidate_aa.to(torch.float).squeeze(-1)
        candidate_aa = candidate_aa.masked_fill(~inference_mask,-float('inf'))
        next_aa_score, next_aa = candidate_aa.max(1,keepdim=True)
        seq = decoder_step_input['sequence_input']['candidate_aa'][0,next_aa]
        seq_mass = decoder_step_input['sequence_input']['seq_mass_forward']+decoder_step_input['sequence_input']['candidate_aa_mass'][0,next_aa]
        seq_pos = decoder_step_input['sequence_input']['seq_pos']
        seq_result = self.candidate_aa[0,next_aa]
        if self.cfg.data.max_iter % 2:
            final_max_iter_ntoc = (self.iter_num == self.cfg.data.max_iter).squeeze(1)
            final_max_iter_cton = (self.iter_num == self.cfg.data.max_iter-1).squeeze(1)
        else:
            final_max_iter_ntoc = (self.iter_num == self.cfg.data.max_iter-1).squeeze(1)
            final_max_iter_cton = (self.iter_num == self.cfg.data.max_iter).squeeze(1)
        self.result_seq_ntoc[final_max_iter_ntoc] = self.result_seq_ntoc[final_max_iter_ntoc].scatter(-1,seq_pos[final_max_iter_ntoc],seq_result[final_max_iter_ntoc])
        self.result_seq_score_ntoc[final_max_iter_ntoc] = self.result_seq_score_ntoc[final_max_iter_ntoc].scatter(-1,seq_pos[final_max_iter_ntoc],next_aa_score[final_max_iter_ntoc])
        self.result_seq_cton[final_max_iter_cton] = self.result_seq_cton[final_max_iter_cton].scatter(-1,seq_pos[final_max_iter_cton],seq_result[final_max_iter_cton])
        self.result_seq_score_cton[final_max_iter_cton] = self.result_seq_score_cton[final_max_iter_cton].scatter(-1,seq_pos[final_max_iter_cton],next_aa_score[final_max_iter_cton])

        seq_pos = seq_pos + 1

        finish_iter_flag = (self.precursor_mass-seq_mass) < 10
        self.iter_num[finish_iter_flag] += 1
        if (self.iter_num>self.cfg.data.max_iter).all():
            return decoder_step_input
        else:
            n_term_inference_flag = (self.iter_num%2).bool()
            seq = torch.where(finish_iter_flag & n_term_inference_flag, 1, seq)
            seq = torch.where(finish_iter_flag & ~n_term_inference_flag, 2, seq)
            seq_mass = torch.where(finish_iter_flag, 0, seq_mass)
            seq_pos = torch.where(finish_iter_flag, 0, seq_pos)

            decoder_step_input['sequence_input']['seq'] = seq
            decoder_step_input['sequence_input']['seq_pos'] = seq_pos
            decoder_step_input['sequence_input']['seq_iter'] = self.iter_num
            decoder_step_input['sequence_input']['seq_mass_forward'] = seq_mass
            decoder_step_input['cache_seqlens'] += 1
            return decoder_step_input

    def result_generator(self):
        score_max_flag = self.result_seq_score_ntoc.sum(1,keepdim=True)>self.result_seq_score_cton.sum(1,keepdim=True)
        result_seq = torch.where(score_max_flag,self.result_seq_ntoc,self.result_seq_cton).cpu().tolist()
        result_seq_score = torch.where(score_max_flag,self.result_seq_score_ntoc,self.result_seq_score_cton).cpu().tolist()

        result = [[] for _ in range(self.result_seq_ntoc.size(0))]
        result_score = [[] for _ in range(self.result_seq_ntoc.size(0))]
        for i in range(self.result_seq_ntoc.size(0)):
            result_seq_row = result_seq[i]
            result_seq_score_row = result_seq_score[i]
            result_seq_row = [aa for aa in result_seq_row if aa!=0]
            result_seq_score_row = [aa for aa in result_seq_score_row if aa!=0]
            if not score_max_flag[i]:
                result_seq_row = result_seq_row[::-1]
                result_seq_score_row = result_seq_score_row[::-1]
            for aa, aa_score in zip(result_seq_row,result_seq_score_row):
                result[i] += [self.candidate_ptm_aa[aa]]
                result_score[i] += [aa_score]
        return result, result_score


def _parse_dynamic_candidate_aa(aa, candidate_ptm_aa):
    if '[' not in aa or not aa.endswith(']'):
        raise ValueError(
            f"Unsupported SeqFiller candidate amino acid {aa!r}. Expected a built-in "
            "residue token such as 'C|UniMod:4' or a dynamic mass token such as 'T[57.021464]'."
        )
    residue, raw_delta = aa.split('[', 1)
    raw_delta = raw_delta[:-1]
    if len(residue) != 1 or residue not in candidate_ptm_aa:
        raise ValueError(f"Unsupported SeqFiller candidate residue in {aa!r}")
    try:
        delta_mass = float(raw_delta)
    except ValueError as exc:
        raise ValueError(f"Dynamic SeqFiller candidate {aa!r} has non-numeric mass delta") from exc
    return residue, delta_mass
