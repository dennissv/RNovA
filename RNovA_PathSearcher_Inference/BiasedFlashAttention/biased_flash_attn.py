# Copyright (c) 2024 Zeping Mao
#
# This code is authored by Zeping Mao. For inquiries, please contact z37mao@uwaterloo.ca.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton
from .biased_flash_attention_fwd import biased_flash_attn_fwd
from .biased_flash_attention_bac import delta_calculation, biased_flash_attn_dkvb, biased_flash_attn_dq

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from math import sqrt

class BiasedFlashAttn(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda',cast_inputs=torch.float16)
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                b: torch.Tensor,
                sm_scale: float = None):
        # Validate the shape of the input matrix
        assert q.dim() == k.dim() == v.dim() == b.dim() == 4
        batch_size, seq_len_q, rel_size, key_size = q.shape
        seq_len_kv = k.size(1)
        assert k.shape==v.shape
        assert k.size(0)==batch_size and k.size(2)==rel_size and k.size(3)==key_size
        assert b.size(0)==batch_size and b.size(1)==seq_len_q and b.size(2)==seq_len_kv and b.size(3)==rel_size

        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        b = b.permute(0,3,1,2)

        # Validate the type of the input matrix
        '''if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype('cuda')
            if q.dtype==torch.float32: q = q.to(autocast_dtype)
            if k.dtype==torch.float32: k = k.to(autocast_dtype)
            if v.dtype==torch.float32: v = v.to(autocast_dtype)
            if b.dtype==torch.float32: b = b.to(autocast_dtype)'''
        assert q.dtype == k.dtype == v.dtype == b.dtype
        assert q.dtype == torch.float16 or q.dtype==torch.bfloat16

        if sm_scale==None: sm_scale = 1/sqrt(key_size)
        m = torch.zeros(batch_size, rel_size, seq_len_q, dtype=q.dtype, device=q.device)
        o = torch.empty(batch_size, rel_size, seq_len_q, key_size, device=q.device, dtype=q.dtype)
        grid = lambda meta: (triton.cdiv(seq_len_q, meta['BLOCK_Q']), batch_size*rel_size)
        biased_flash_attn_fwd[grid](q,k,v,b,o,m,
                                    q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                                    k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                                    v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                                    b.stride(0),b.stride(1),b.stride(2),b.stride(3),
                                    o.stride(0),o.stride(1),o.stride(2),o.stride(3),
                                    m.stride(0),m.stride(1),m.stride(2),
                                    seq_len_q, seq_len_kv,
                                    SM_SCALE=sm_scale,
                                    REL_SIZE=rel_size, KEY_SIZE=key_size)
        ctx.save_for_backward(q,k,v,b,m,o)
        ctx.batch_size = batch_size
        ctx.rel_size = rel_size
        ctx.seq_len_q = seq_len_q
        ctx.seq_len_kv = seq_len_kv
        ctx.key_size = key_size
        ctx.sm_scale = sm_scale
        o = o.permute(0,2,1,3).flatten(2,3)
        return o

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, do: torch.Tensor):
        q,k,v,b,m,o = ctx.saved_tensors
        batch_size = ctx.batch_size
        rel_size = ctx.rel_size
        seq_len_q = ctx.seq_len_q
        seq_len_kv = ctx.seq_len_kv
        key_size = ctx.key_size
        sm_scale = ctx.sm_scale
        do = do.reshape(batch_size, seq_len_q, rel_size, key_size).permute(0,2,1,3)

        delta = torch.zeros(batch_size, rel_size, seq_len_q, device=o.device, dtype=o.dtype)
        grid = lambda meta: (triton.cdiv(seq_len_q, meta['BLOCK_Q']), batch_size*rel_size)
        delta_calculation[grid](o, do, delta,
                                o.stride(0),o.stride(1),o.stride(2),o.stride(3),
                                do.stride(0),do.stride(1),do.stride(2),do.stride(3),
                                delta.stride(0),delta.stride(1),delta.stride(2),
                                seq_len_q,
                                REL_SIZE=rel_size,
                                KEY_SIZE=key_size)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        db = torch.zeros_like(b)
        grid = lambda meta: (triton.cdiv(seq_len_kv, meta['BLOCK_K']), batch_size*rel_size)
        biased_flash_attn_dkvb[grid](q, k, v, b, m, delta, # FlashAttn Input
                                     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                                     b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                                     m.stride(0), m.stride(1), m.stride(2),
                                     delta.stride(0), delta.stride(1), delta.stride(2),
                                     do, # FlashAttn Gradient Output
                                     do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                                     dk,dv,db, # Gradient Backpropogation
                                     dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                                     dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                                     db.stride(0), db.stride(1), db.stride(2), db.stride(3),
                                     seq_len_q, seq_len_kv,
                                     SM_SCALE=sm_scale,
                                     REL_SIZE=rel_size,
                                     KEY_SIZE=key_size)

        grid = lambda meta: (triton.cdiv(seq_len_q, meta['BLOCK_Q']), batch_size*rel_size)
        biased_flash_attn_dq[grid](k, db,
                                   k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                   db.stride(0), db.stride(1), db.stride(2), db.stride(3),
                                   dq,
                                   dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                                   seq_len_q, seq_len_kv,
                                   SM_SCALE=sm_scale,
                                   REL_SIZE=rel_size,
                                   KEY_SIZE=key_size)

        dq = dq.permute(0,2,1,3)
        dk = dk.permute(0,2,1,3)
        dv = dv.permute(0,2,1,3)
        db = db.permute(0,2,3,1)
        return dq, dk, dv, db, None

biased_flash_attn_func = BiasedFlashAttn.apply

# Unit Test
if __name__=='__main__':
    batch_size, rel_size, seq_len_q, seq_len_kv, key_size = 8, 32, 119, 130, 64
    sm_scale = 0.5
    for i in range(100):
        q_flash = torch.randn(batch_size, seq_len_q, rel_size, key_size, device='cuda', dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)
        k_flash = torch.randn(batch_size, seq_len_kv, rel_size, key_size, device='cuda', dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)
        v_flash = torch.randn(batch_size, seq_len_kv, rel_size, key_size, device='cuda', dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)
        b_flash = torch.randn(batch_size, seq_len_q, seq_len_kv, rel_size, device='cuda', dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)
        dout = torch.randn_like(q_flash).flatten(2,3)

        q_torch = q_flash.clone().requires_grad_()
        k_torch = k_flash.clone().requires_grad_()
        v_torch = v_flash.clone().requires_grad_()
        b_torch = b_flash.clone().requires_grad_()

        q_flash.requires_grad_()
        k_flash.requires_grad_()
        v_flash.requires_grad_()
        b_flash.requires_grad_()

        qk_torch = torch.einsum('bqrh,bkrh->bqkr',q_torch,k_torch) * sm_scale + b_torch
        o_torch = torch.einsum('bqkr,bkrh->bqrh',qk_torch.softmax(-2),v_torch).flatten(2,3)
        o_torch.backward(dout)
        dq_torch = q_torch.grad
        dk_torch = k_torch.grad
        dv_torch = v_torch.grad
        db_torch = b_torch.grad

        o_flash = biased_flash_attn_func(q_flash,k_flash,v_flash,b_flash,sm_scale)
        o_flash.backward(dout)
        dq_flash = q_flash.grad
        dk_flash = k_flash.grad
        dv_flash = v_flash.grad
        db_flash = b_flash.grad

        print(i,
              (o_torch-o_flash).abs().max().item(),
              (dq_torch-dq_flash).abs().max().item(),
              (dk_torch-dk_flash).abs().max().item(),
              (dv_torch-dv_flash).abs().max().item(),
              (db_torch-db_flash).abs().max().item(),)
