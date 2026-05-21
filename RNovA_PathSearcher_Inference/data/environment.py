import torch
import numpy as np

from math import ceil

class Environment(object):
    def __init__(self, cfg, model, inference_dl, device, logger=None, progress_interval=10, cache_policy="auto"):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.device = device
        self.logger = logger
        self.progress_interval = progress_interval
        self.cache_policy = cache_policy
        self.generator = torch.Generator('cuda')
        self.generator.manual_seed(0)
        kernel_size = 3
        self.kernel = np.ones_like(kernel_size)

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        decoder_step_input, node, next_node_mask, title, node_mass_output = self.exploration_initializing()
        decoder_step_input, next_node_mask = self.next_aa_choice(node, next_node_mask, decoder_step_input)

        decoder_steps = 1
        while next_node_mask!=None:
            if self.logger and decoder_steps % self.progress_interval == 0:
                self.logger.debug(
                    "PathSearcher decoder step %s: active_spectra=%s, cuda_allocated=%.2f GiB, cuda_reserved=%.2f GiB",
                    decoder_steps,
                    int(self.remain_index.numel()),
                    torch.cuda.memory_allocated(self.device) / 1024**3,
                    torch.cuda.memory_reserved(self.device) / 1024**3,
                )
            node = self.model(**decoder_step_input)
            decoder_step_input, next_node_mask = self.next_aa_choice(node, next_node_mask, decoder_step_input)
            decoder_steps += 1

        #nterm_node_seq_list, nterm_score_seq_list, nterm_class_seq_list, cterm_node_seq_list, cterm_score_seq_list, cterm_class_seq_list = self.result_generation()
        #return nterm_node_seq_list, nterm_score_seq_list, nterm_class_seq_list, cterm_node_seq_list, cterm_score_seq_list, cterm_class_seq_list, title, node_mass_output.cpu().numpy()
        node_seq_list, score_seq_list, class_seq_list = self.result_generation()
        if self.logger:
            self.logger.debug("PathSearcher batch completed after %s decoder step(s)", decoder_steps)
        return node_seq_list, score_seq_list, class_seq_list, title, node_mass_output.cpu().numpy()

    def exploration_initializing(self):
        node_input, node_mask, title = next(self.inference_dl)
        if self.logger:
            self.logger.debug(
                "PathSearcher batch received: spectra=%s, padded_nodes=%s, cuda_allocated=%.2f GiB, cuda_reserved=%.2f GiB",
                node_mask.size(0),
                node_mask.size(1),
                torch.cuda.memory_allocated(self.device) / 1024**3,
                torch.cuda.memory_reserved(self.device) / 1024**3,
            )
        self.node_class = node_input['node_class']
        self.node_mass = node_input['node_mass']
        self.node_last_mass = self.node_mass[self.node_class==3].cpu().numpy()
        node_mass_output = node_input['node_mass'].clone()
        self.node_mask = node_mask
        self.node_last_index = node_mask.cumsum(1).max(1,keepdim=True).values

        # Input prepration.
        self.node_index = torch.ones([node_mask.size(0),1],dtype=torch.long,device=self.device)
        self.node_index_pos = torch.zeros_like(self.node_index)
        self.iter_num = torch.ones_like(self.node_index)

        max_cache_seq_len = self._max_cache_seq_len()
        self.remain_index = torch.arange(node_mask.size(0),device=self.device).unsqueeze(1)
        self.result_cache = -torch.ones([node_mask.size(0), max_cache_seq_len], dtype=torch.float, device=self.device)
        self.result_score_cache = torch.zeros([node_mask.size(0), max_cache_seq_len], dtype=torch.half, device=self.device)
        self.result_class_cache = torch.zeros([node_mask.size(0), max_cache_seq_len], dtype=torch.long, device=self.device)
        self.result_iter_cache = -torch.ones([node_mask.size(0), max_cache_seq_len], dtype=torch.long, device=self.device)

        self.decoder_k_cache = torch.zeros(node_mask.size(0),
                                           self.cfg.model.decoder.num_layers,
                                           max_cache_seq_len,
                                           self.cfg.model.decoder.num_heads,
                                           self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads,
                                           dtype=torch.float16,
                                           device=self.device)
        self.decoder_v_cache = torch.zeros(node_mask.size(0),
                                           self.cfg.model.decoder.num_layers,
                                           max_cache_seq_len,
                                           self.cfg.model.decoder.num_heads,
                                           self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads,
                                           dtype=torch.float16,
                                           device=self.device)

        if self.logger:
            cache_gib = (
                self.decoder_k_cache.numel() * self.decoder_k_cache.element_size()
                + self.decoder_v_cache.numel() * self.decoder_v_cache.element_size()
            ) / 1024**3
            self.logger.debug(
                "PathSearcher decoder cache allocated: %.2f GiB, max_cache_seq_len=%s",
                cache_gib,
                max_cache_seq_len,
            )
            self.logger.debug("PathSearcher encoder forward starting")
        k_cache, v_cache, node_embedding = self.model.encoder_forward(**node_input)
        node_embedding_output = self.model.prepare_node_embedding_output(node_embedding)
        if self.logger:
            self.logger.debug("PathSearcher encoder forward finished; initial decoder forward starting")

        decoder_step_input = {
            'node_index': self.node_index,
            'node_index_pos': self.node_index_pos,
            'node_index_iter': self.iter_num,
            'cache_seqlens': 0,
            # Constant Value for a batch
            'node_embedding': node_embedding,
            'node_embedding_output': node_embedding_output,
            'k_cache': k_cache,
            'v_cache': v_cache,
            'decoder_k_cache': self.decoder_k_cache,
            'decoder_v_cache': self.decoder_v_cache
        }
        node = self.model(**decoder_step_input)
        if self.logger:
            self.logger.debug("PathSearcher initial decoder forward finished")
        next_node_mask = self.step_label_generation(self.node_index)
        return decoder_step_input, node, next_node_mask, title, node_mass_output

    def next_aa_choice(self, node, next_node_mask, decoder_step_input):
        if decoder_step_input['cache_seqlens'] >= self.result_cache.size(1):
            raise RuntimeError(
                "PathSearcher decoder cache was exhausted; rerun with --path-cache-policy legacy"
            )
        node = node.squeeze(1)
        node = node.masked_fill(~next_node_mask, -float('inf'))
        self.node_index = node.argmax(1,keepdim=True)
        self.node_index_pos += 1
        self.result_cache[self.remain_index, decoder_step_input['cache_seqlens']] = self.node_mass.gather(1,self.node_index)
        self.result_score_cache[self.remain_index, decoder_step_input['cache_seqlens']] = node.gather(1,self.node_index)
        self.result_class_cache[self.remain_index, decoder_step_input['cache_seqlens']] = self.node_class.gather(1,self.node_index)
        self.result_iter_cache[self.remain_index, decoder_step_input['cache_seqlens']] = self.iter_num

        n_term_inference_flag = (self.iter_num%2).bool()
        next_node_mask = self.step_label_generation(self.node_index)
        finish_iter_flag = torch.logical_or(self.node_index_pos>self.cfg.data.peptide_max_len, ~next_node_mask.any(1, keepdim=True))
        self.iter_num[finish_iter_flag] += 1

        finish_explore_flag = (self.iter_num>self.cfg.data.max_iter).squeeze(1)
        node = node[~finish_explore_flag]
        finish_iter_flag = finish_iter_flag[~finish_explore_flag]
        self.remain_index = self.remain_index[~finish_explore_flag]
        self.node_index = self.node_index[~finish_explore_flag]
        self.node_class = self.node_class[~finish_explore_flag]
        self.node_index_pos = self.node_index_pos[~finish_explore_flag]
        self.iter_num = self.iter_num[~finish_explore_flag]
        self.node_mass = self.node_mass[~finish_explore_flag]
        self.node_mask = self.node_mask[~finish_explore_flag]
        self.node_last_index = self.node_last_index[~finish_explore_flag]
        decoder_step_input['node_embedding'] = decoder_step_input['node_embedding'][~finish_explore_flag]
        decoder_step_input['node_embedding_output'] = decoder_step_input['node_embedding_output'][~finish_explore_flag]
        decoder_step_input['k_cache'] = [k[~finish_explore_flag] for k in decoder_step_input['k_cache']]
        decoder_step_input['v_cache'] = [v[~finish_explore_flag] for v in decoder_step_input['v_cache']]
        decoder_step_input['decoder_k_cache'] = decoder_step_input['decoder_k_cache'][~finish_explore_flag]
        decoder_step_input['decoder_v_cache'] = decoder_step_input['decoder_v_cache'][~finish_explore_flag]

        if finish_explore_flag.all():
            decoder_step_input['cache_seqlens'] += 1
            return decoder_step_input, None
        else:
            n_term_inference_flag = (self.iter_num%2).bool()
            self.node_index = torch.where(finish_iter_flag & n_term_inference_flag, 1, self.node_index)
            self.node_index = torch.where(finish_iter_flag & ~n_term_inference_flag, self.node_last_index, self.node_index)
            self.node_index_pos = torch.where(finish_iter_flag, 0, self.node_index_pos)

            decoder_step_input['node_index'] = self.node_index
            decoder_step_input['node_index_pos'] = self.node_index_pos
            decoder_step_input['node_index_iter'] = self.iter_num

            next_node_mask = self.step_label_generation(self.node_index)
            #self.result_cache[self.remain_index, decoder_step_input['cache_seqlens']] = self.node_mass.gather(1,self.node_index)
            #self.result_score_cache[self.remain_index, decoder_step_input['cache_seqlens']] = node.gather(1,self.node_index)
            #self.result_iter_cache[self.remain_index, decoder_step_input['cache_seqlens']] = self.iter_num
            decoder_step_input['cache_seqlens'] += 1
            return decoder_step_input, next_node_mask

    def _max_cache_seq_len(self):
        if self.cache_policy == "legacy":
            return ceil((self.cfg.data.peptide_max_len+20)*self.cfg.data.max_iter*10)
        return ceil((self.cfg.data.peptide_max_len + 2) * self.cfg.data.max_iter + 8)

    def result_generation(self):
        node_seq_list, score_seq_list, class_seq_list = [], [], []
        nterm_node_seq_list, nterm_score_seq_list, nterm_class_seq_list = [], [], []
        cterm_node_seq_list, cterm_score_seq_list, cterm_class_seq_list = [], [], []
        if self.cfg.data.max_iter==1:
            mask = self.result_cache!=-1
            for i in range(mask.size(0)):
                node_seq_list.append(np.pad(self.result_cache[i][mask[i]].cpu().numpy(),(1,0),constant_values=0))
                score_seq_list.append(np.pad(self.result_score_cache[i][mask[i]].cpu().numpy(),(1,0),constant_values=10))
                class_seq_list.append(np.pad(self.result_class_cache[i][mask[i]].cpu().numpy(),(1,0),constant_values=2))
            return node_seq_list, score_seq_list, class_seq_list
        elif self.cfg.data.max_iter%2==0:
            mask = self.result_cache!=-1
            for i in range(mask.size(0)):
                precursor_mass = self.node_last_mass[i]
                result_ntoc = self.result_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter-1,mask[i])].cpu().numpy()
                result_cton = self.result_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter,mask[i])].cpu().numpy()[::-1]
                result_score_ntoc = self.result_score_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter-1,mask[i])].cpu().numpy()
                result_score_cton = self.result_score_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter,mask[i])].cpu().numpy()[::-1]
                result_class_ntoc = self.result_class_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter-1,mask[i])].cpu().numpy()
                result_class_cton = self.result_class_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter,mask[i])].cpu().numpy()[::-1]
                node_seq, score_seq, class_seq = self.result_combiner(result_ntoc,result_cton,result_score_ntoc,result_score_cton,result_class_ntoc,result_class_cton,precursor_mass)
                node_seq_list.append(node_seq)
                score_seq_list.append(score_seq)
                class_seq_list.append(class_seq)

                if result_class_ntoc[-1]==3:
                    result_ntoc = np.pad(result_ntoc, (1,0), constant_values=0)
                    result_score_ntoc = np.pad(result_score_ntoc, (1,0), constant_values=10)
                    result_class_ntoc = np.pad(result_class_ntoc, (1,0), constant_values=2)
                else:
                    result_ntoc = np.pad(result_ntoc, (1,1), constant_values=(0,precursor_mass))
                    result_score_ntoc = np.pad(result_score_ntoc, (1,1), constant_values=(10,-10))
                    result_class_ntoc = np.pad(result_class_ntoc, (1,1), constant_values=(2,3))

                if result_class_cton[0]==2:
                    result_cton = np.pad(result_cton, (0,1), constant_values=precursor_mass)
                    result_score_cton = np.pad(result_score_cton, (0,1), constant_values=10)
                    result_class_cton = np.pad(result_class_cton, (0,1), constant_values=3)
                else:
                    result_cton = np.pad(result_cton, (1,1), constant_values=(0,precursor_mass))
                    result_score_cton = np.pad(result_score_cton, (1,1), constant_values=(-10,10))
                    result_class_cton = np.pad(result_class_cton, (1,1), constant_values=(2,3))

                nterm_node_seq_list.append(result_ntoc)
                nterm_score_seq_list.append(result_score_ntoc)
                nterm_class_seq_list.append(result_class_ntoc)
                cterm_node_seq_list.append(result_cton)
                cterm_score_seq_list.append(result_score_cton)
                cterm_class_seq_list.append(result_class_cton)
            #return nterm_node_seq_list, nterm_score_seq_list, nterm_class_seq_list, cterm_node_seq_list, cterm_score_seq_list, cterm_class_seq_list
            return node_seq_list, score_seq_list, class_seq_list
        elif self.cfg.data.max_iter%2==1:
            mask = self.result_cache!=-1
            for i in range(mask.size(0)):
                precursor_mass = self.node_last_mass[i]
                result_ntoc = self.result_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter,mask[i])].cpu().numpy()
                result_cton = self.result_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter-1,mask[i])].cpu().numpy()[::-1]
                result_score_ntoc = self.result_score_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter,mask[i])].cpu().numpy()
                result_score_cton = self.result_score_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter-1,mask[i])].cpu().numpy()[::-1]
                result_class_ntoc = self.result_class_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter,mask[i])].cpu().numpy()
                result_class_cton = self.result_class_cache[i][torch.logical_and(self.result_iter_cache[i]==self.cfg.data.max_iter-1,mask[i])].cpu().numpy()[::-1]
                node_seq, score_seq, class_seq = self.result_combiner(result_ntoc,result_cton,result_score_ntoc,result_score_cton,result_class_ntoc,result_class_cton,precursor_mass)
                node_seq_list.append(node_seq)
                score_seq_list.append(score_seq)
                class_seq_list.append(class_seq)

                if result_class_ntoc[-1]==3:
                    result_ntoc = np.pad(result_ntoc, (1,0), constant_values=0)
                    result_score_ntoc = np.pad(result_score_ntoc, (1,0), constant_values=10)
                    result_class_ntoc = np.pad(result_class_ntoc, (1,0), constant_values=2)
                else:
                    result_ntoc = np.pad(result_ntoc, (1,1), constant_values=(0,precursor_mass))
                    result_score_ntoc = np.pad(result_score_ntoc, (1,1), constant_values=(10,-10))
                    result_class_ntoc = np.pad(result_class_ntoc, (1,1), constant_values=(2,3))

                if result_class_cton[0]==2:
                    result_cton = np.pad(result_cton, (0,1), constant_values=precursor_mass)
                    result_score_cton = np.pad(result_score_cton, (0,1), constant_values=10)
                    result_class_cton = np.pad(result_class_cton, (0,1), constant_values=3)
                else:
                    result_cton = np.pad(result_cton, (1,1), constant_values=(0,precursor_mass))
                    result_score_cton = np.pad(result_score_cton, (1,1), constant_values=(-10,10))
                    result_class_cton = np.pad(result_class_cton, (1,1), constant_values=(2,3))

                nterm_node_seq_list.append(result_ntoc)
                nterm_score_seq_list.append(result_score_ntoc)
                nterm_class_seq_list.append(result_class_ntoc)
                cterm_node_seq_list.append(result_cton)
                cterm_score_seq_list.append(result_score_cton)
                cterm_class_seq_list.append(result_class_cton)
            #return nterm_node_seq_list, nterm_score_seq_list, nterm_class_seq_list, cterm_node_seq_list, cterm_score_seq_list, cterm_class_seq_list
            return node_seq_list, score_seq_list, class_seq_list
        elif self.cfg.data.max_iter<=0:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def result_combiner(self,result_ntoc,result_cton,result_score_ntoc,result_score_cton,result_class_ntoc,result_class_cton,precursor_mass):
        if result_class_ntoc[-1]==3 and result_class_cton[0]==2:
            result_ntoc = result_ntoc[:-1]
            result_score_ntoc = result_score_ntoc[:-1]
            result_class_ntoc = result_class_ntoc[:-1]

            result_cton = result_cton[1:]
            result_score_cton = result_score_cton[1:]
            result_class_cton = result_class_cton[1:]

            i = 0
            j = 0
            result_ntoc_match_index = []
            result_cton_match_index = []
            while i < len(result_ntoc) and j < len(result_cton):
                if abs(result_ntoc[i] - result_cton[j]) < 0.01:
                    result_ntoc_match_index.append(i)
                    result_cton_match_index.append(j)
                    i += 1
                    j += 1
                elif result_ntoc[i] < result_cton[j]:
                    i += 1
                else:
                    j += 1
            result_ntoc_match_index = np.array(result_ntoc_match_index)
            result_cton_match_index = np.array(result_cton_match_index)
            if result_ntoc_match_index.size<3:
                if result_score_ntoc.sum()>result_score_cton.sum():
                    node_seq = result_ntoc
                    score_seq = result_score_ntoc
                    class_seq = result_class_ntoc
                else:
                    node_seq = result_cton
                    score_seq = result_score_cton
                    class_seq = result_class_cton
            else:
                result_score_ntoc = np.convolve(result_score_ntoc, self.kernel, 'same')
                result_score_cton = np.convolve(result_score_cton, self.kernel, 'same')
                result_score_ntoc_match = result_score_ntoc[result_ntoc_match_index]
                result_score_cton_match = result_score_cton[result_cton_match_index]
                if (result_score_ntoc_match>result_score_cton_match).all():
                    node_seq = result_ntoc
                    score_seq = result_score_ntoc
                    class_seq = result_class_ntoc
                else:
                    seperate_index = np.where(result_score_ntoc_match<=result_score_cton_match)[0][0]
                    result_ntoc_match_index = result_ntoc_match_index[seperate_index]
                    result_cton_match_index = result_cton_match_index[seperate_index]
                    node_seq = np.concatenate([result_ntoc[:result_ntoc_match_index],result_cton[result_cton_match_index:]])
                    score_seq = np.concatenate([result_score_ntoc[:result_ntoc_match_index],result_score_cton[result_cton_match_index:]])
                    class_seq = np.concatenate([result_class_ntoc[:result_ntoc_match_index],result_class_cton[result_cton_match_index:]])
            node_seq = np.pad(node_seq, (1,1), constant_values=(0,precursor_mass))
            score_seq = np.pad(score_seq, (1,1), constant_values=(10,10))
            class_seq = np.pad(class_seq, (1,1), constant_values=(2,3))
        elif result_class_ntoc[-1]==3 and result_class_cton[0]!=2:
            node_seq = np.pad(result_ntoc, (1,0), constant_values=0)
            score_seq = np.pad(result_score_ntoc, (1,0), constant_values=10)
            score_seq[-1] = 10
            class_seq = np.pad(result_class_ntoc, (1,0), constant_values=2)
        elif result_class_ntoc[-1]!=3 and result_class_cton[0]==2:
            node_seq = np.pad(result_cton, (0,1), constant_values=precursor_mass)
            score_seq = np.pad(result_score_cton, (0,1), constant_values=10)
            score_seq[0] = 10
            class_seq = np.pad(result_class_cton, (0,1), constant_values=3)
        else:
            if result_score_ntoc.sum()>result_score_cton.sum():
                node_seq = np.pad(result_ntoc, (1,0), constant_values=0)
                score_seq = np.pad(result_score_ntoc, (1,0), constant_values=10)
                class_seq = np.pad(result_class_ntoc, (1,0), constant_values=2)
            else:
                node_seq = np.pad(result_cton, (0,1), constant_values=precursor_mass)
                score_seq = np.pad(result_score_cton, (0,1), constant_values=10)
                class_seq = np.pad(result_class_cton, (0,1), constant_values=3)
        return node_seq, score_seq, class_seq

    def step_label_generation(self, node_index):
        n_term_inference_flag = (self.iter_num%2).bool()
        current_mass = self.node_mass.gather(index=node_index,dim=1)
        # Magic Number 53.997988624 is the mass of backbone of amino acid. Composition NCCO.
        # To consider the mass error of the mass spec. We use the mass error of 1 Da.
        next_node_mask = torch.where(n_term_inference_flag,self.node_mass>current_mass+53,self.node_mass<current_mass-53)
        next_node_mask = torch.logical_and(next_node_mask,self.node_mask)
        return next_node_mask
