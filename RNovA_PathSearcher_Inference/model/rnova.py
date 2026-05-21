import torch
from torch import nn
from .encoderlayer import EncoderLayer
from .decoderlayer import DecoderLayer
from .node_embedding import NodeEmbedding
from .path_embedding import PathEmbedding

class Decoder_Cross_Projector(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.decoder_cross_kv = nn.Linear(cfg.model.hidden_size,
                                          cfg.model.hidden_size*cfg.model.decoder.num_layers*2)

    def forward(self, node):
        k_cache, v_cache = self.decoder_cross_kv(node).view(node.size(0),-1,
                                                            self.cfg.model.decoder.num_heads*self.cfg.model.decoder.num_layers*2,
                                                            self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads).chunk(2,dim=-2)
        k_cache = k_cache/torch.linalg.vector_norm(k_cache,dim=-1,keepdim=True)
        return k_cache.transpose(1,2).chunk(self.cfg.model.decoder.num_layers, dim=1), v_cache.transpose(1,2).chunk(self.cfg.model.decoder.num_layers, dim=1)

class RNovA(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.node_embedding_layer = NodeEmbedding(cfg)
        self.path_embedding_layer = PathEmbedding(cfg)
        self.node_decoder_embedding = Decoder_Cross_Projector(cfg)
        self.encoder = nn.ModuleList([EncoderLayer(cfg.model.hidden_size,
                                                   cfg.model.encoder.rel_size,
                                                   cfg.model.encoder.key_size,
                                                   cfg.model.encoder.rel_feature_num) \
                                                   for _ in range(cfg.model.encoder.num_layers)])

        self.node_index_pos_embedding_layer = nn.Embedding(cfg.data.peptide_max_len+1, cfg.model.hidden_size)
        self.node_index_iter_embedding_layer = nn.Embedding(cfg.data.max_iter*40, cfg.model.hidden_size)
        self.decoder = nn.ModuleList([DecoderLayer(cfg.model.hidden_size,
                                                   cfg.model.decoder.num_heads) \
                                                   for _ in range(cfg.model.decoder.num_layers)])
        self.output_decoder = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)
        self.output_encoder = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)

    @torch.no_grad()
    @torch.autocast('cuda',dtype=torch.float16)
    def encoder_forward(self, node_mass, peak_intensity_rank, node_intensity, charge, node_class, peak_moverz):
        node, node_relative_pos = self.node_embedding_layer(node_mass, peak_intensity_rank, node_intensity, charge, node_class, peak_moverz)
        path = self.path_embedding_layer(node_mass)
        for encoder_layer in self.encoder: node = encoder_layer(node, node_relative_pos, path)
        k_cache, v_cache = self.node_decoder_embedding(node)
        return k_cache, v_cache, node

    @torch.no_grad()
    @torch.autocast('cuda',dtype=torch.float16)
    def prepare_node_embedding_output(self, node_embedding):
        node_embedding = self.output_encoder(node_embedding)
        return node_embedding/torch.linalg.vector_norm(node_embedding,dim=-1,keepdim=True)

    @torch.no_grad()
    @torch.autocast('cuda',dtype=torch.float16)
    def forward(self, node_index, node_index_pos, node_index_iter, node_embedding, cache_seqlens, k_cache, v_cache, decoder_k_cache, decoder_v_cache, node_embedding_output=None):
        node_index_seq = node_index.unsqueeze(-1).repeat_interleave(node_embedding.size(-1),dim=-1)
        node = node_embedding.gather(dim=1,index=node_index_seq)
        node_index_pos = self.node_index_pos_embedding_layer(node_index_pos)
        node_index_iter = self.node_index_iter_embedding_layer(node_index_iter)
        node = node + node_index_pos + node_index_iter
        for i, decoder_layer in enumerate(self.decoder):
            node = decoder_layer(node,
                                 cache_seqlens,
                                 k_cache[i],v_cache[i],
                                 decoder_k_cache[:,i],decoder_v_cache[:,i])
        node = self.output_decoder(node)
        if node_embedding_output is None:
            node_embedding_output = self.prepare_node_embedding_output(node_embedding)
        node = node @ node_embedding_output.transpose(1,2)
        return node
