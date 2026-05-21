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

import torch
from math import sqrt

import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_Q': BQ, 'BLOCK_K': BK}, num_stages=s, num_warps=w) \
    for BQ in [32, 64, 128, 256]\
    for BK in [32, 64, 128]\
    for s in ([3, 4, 7])\
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_Q = conf.kwargs["BLOCK_Q"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    if BLOCK_Q * BLOCK_K < 128 * 256 and conf.num_warps == 8:
        return False
    return True

@triton.autotune(list(filter(keep, configs)),
                 key=["seq_len_q","seq_len_kv"])
@triton.jit
def biased_flash_attn_fwd(Q, K, V, B, O, M,
                          stride_qb, stride_qr, stride_qq, stride_qh,
                          stride_kb, stride_kr, stride_kk, stride_kh,
                          stride_vb, stride_vr, stride_vk, stride_vh,
                          stride_bb, stride_br, stride_bq, stride_bk,
                          stride_ob, stride_or, stride_oq, stride_oh,
                          stride_mb, stride_mr, stride_mq,
                          seq_len_q, seq_len_kv,
                          SM_SCALE: tl.constexpr,
                          REL_SIZE: tl.constexpr,
                          KEY_SIZE: tl.constexpr,
                          BLOCK_Q: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    start_q = tl.program_id(0)
    off_br = tl.program_id(1)
    off_b = off_br // REL_SIZE
    off_r = off_br % REL_SIZE
    q_offset = off_b.to(tl.int64) * stride_qb + off_r.to(tl.int64) * stride_qr
    k_offset = off_b.to(tl.int64) * stride_kb + off_r.to(tl.int64) * stride_kr
    v_offset = off_b.to(tl.int64) * stride_vb + off_r.to(tl.int64) * stride_vr
    b_offset = off_b.to(tl.int64) * stride_bb + off_r.to(tl.int64) * stride_br
    o_offset = off_b.to(tl.int64) * stride_ob + off_r.to(tl.int64) * stride_or
    m_offset = off_b.to(tl.int64) * stride_mb + off_r.to(tl.int64) * stride_mr

    Q_block_ptr = tl.make_block_ptr(
        base=Q+q_offset,
        shape=(seq_len_q, KEY_SIZE),
        strides=(stride_qq, stride_qh),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, KEY_SIZE),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K+k_offset,
        shape=(KEY_SIZE, seq_len_kv),
        strides=(stride_kh, stride_kk),
        offsets=(0, 0),
        block_shape=(KEY_SIZE, BLOCK_K),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
            base=V+v_offset,
            shape=(seq_len_kv, KEY_SIZE),
            strides=(stride_vk, stride_vh),
            offsets=(0, 0),
            block_shape=(BLOCK_K, KEY_SIZE),
            order=(1, 0),
        )
    B_block_ptr = tl.make_block_ptr(
        base=B+b_offset,
        shape=(seq_len_q, seq_len_kv),
        strides=(stride_bq, stride_bk),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_K),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O+o_offset,
        shape=(seq_len_q, KEY_SIZE),
        strides=(stride_oq, stride_oh),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, KEY_SIZE),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1
    o = tl.zeros([BLOCK_Q, KEY_SIZE], dtype=tl.float32)
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
    o, l_i, m_i = _attn_fwd_inner(o, l_i, m_i, q,
                                  K_block_ptr, V_block_ptr, B_block_ptr,
                                  seq_len_kv,
                                  SM_SCALE=SM_SCALE, BLOCK_K=BLOCK_K)
    # epilogue
    m_i += tl.math.log2(l_i)
    o = o / l_i[:, None]
    offs_m = start_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = offs_m.to(tl.int64)<seq_len_q
    m_ptrs = M + m_offset + offs_m * stride_mq
    tl.store(m_ptrs, m_i.to(M.dtype.element_ty), mask=mask)
    tl.store(O_block_ptr, o.to(O.dtype.element_ty), boundary_check=(0,))

@triton.jit
def _attn_fwd_inner(o, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr, B_block_ptr,#
                    seq_len_kv,
                    SM_SCALE: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    for start_k in range(0, seq_len_kv, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero')
        b = tl.load(B_block_ptr, boundary_check=(0,1), padding_option='zero')
        qk = tl.dot(q, k) * SM_SCALE + b
        qk = qk * 1.44269504

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        o = o * alpha[:, None]
        # update o
        v = tl.load(V_block_ptr)
        o = tl.dot(p, v.to(tl.float32), o)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_K, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_K))
    return o, l_i, m_i

# Unit Test
def biased_flash_attention_fwd_unit_test(q: torch.Tensor,
                                         k: torch.Tensor,
                                         v: torch.Tensor,
                                         b: torch.Tensor,
                                         sm_scale: float = None):
    batch_size, rel_size, seq_len_q, key_size = q.shape
    seq_len_kv = k.size(2)
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
                                REL_SIZE=rel_size, KEY_SIZE=key_size,
                                BLOCK_Q=64, BLOCK_K=32)
    return o, m

if __name__=='__main__':
    batch_size, rel_size, seq_len_q, seq_len_kv, key_size = 32, 16, 119, 218, 64
    sm_scale = 1
    #sm_scale = 1/sqrt(key_size)
    q = torch.randn(batch_size, rel_size, seq_len_q, key_size, dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, rel_size, seq_len_kv, key_size, dtype=torch.float32, device='cuda')
    v = torch.randn(batch_size, rel_size, seq_len_kv, key_size, dtype=torch.float32, device='cuda')
    b = torch.randn(batch_size, rel_size, seq_len_q, seq_len_kv, dtype=torch.float32, device='cuda')
    triton_output, m = biased_flash_attention_fwd_unit_test(q,k,v,b,sm_scale)
    torch_output = torch.einsum('brqh,brkh->brqk',q,k)*sm_scale + b
    torch_output = torch.softmax(torch_output, -1)
    torch_output = torch_output@v
    print((triton_output-torch_output).abs().max().item())
