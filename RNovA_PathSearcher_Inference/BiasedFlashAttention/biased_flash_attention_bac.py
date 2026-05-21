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

import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_Q': BQ}, num_stages=s, num_warps=w) \
    for BQ in [32, 64, 128, 256]\
    for s in ([3, 4, 7])\
    for w in [4, 8]\
]

@triton.autotune(configs,
                 key=["seq_len_q"])
@triton.jit
def delta_calculation(O, DO, D,
                      stride_ob, stride_or, stride_oq, stride_oh,
                      stride_dob, stride_dor, stride_doq, stride_doh,
                      stride_db, stride_dr, stride_dq,
                      seq_len_q,
                      BLOCK_Q: tl.constexpr,
                      REL_SIZE: tl.constexpr,
                      KEY_SIZE: tl.constexpr):
    start_q = tl.program_id(0)
    off_br = tl.program_id(1)
    off_b = off_br // REL_SIZE
    off_r = off_br % REL_SIZE
    o_offset = off_b.to(tl.int64) * stride_ob + off_r.to(tl.int64) * stride_or
    do_offset = off_b.to(tl.int64) * stride_dob + off_r.to(tl.int64) * stride_dor
    d_offset = off_b.to(tl.int64) * stride_db + off_r.to(tl.int64) * stride_dr

    O_block_ptr = tl.make_block_ptr(
        base=O+o_offset,
        shape=(seq_len_q, KEY_SIZE),
        strides=(stride_oq, stride_oh),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, KEY_SIZE),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO+do_offset,
        shape=(seq_len_q, KEY_SIZE),
        strides=(stride_doq, stride_doh),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, KEY_SIZE),
        order=(1, 0),
    )

    dq_qblock_index = start_q*BLOCK_Q + tl.arange(0,BLOCK_Q)
    mask_d_ptr = dq_qblock_index.to(tl.int64)<seq_len_q
    D_ptr = D + d_offset + dq_qblock_index * stride_dq

    o = tl.load(O_block_ptr, boundary_check=(0,), padding_option='zero')
    do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option='zero')
    d = tl.sum(o*do, axis=1)
    tl.store(D_ptr, d.to(D.dtype.element_ty), mask=mask_d_ptr)

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
def biased_flash_attn_dkvb(Q, K, V, B, M, D, # FlashAttn Input
                           stride_qb, stride_qr, stride_qq, stride_qh,
                           stride_kb, stride_kr, stride_kk, stride_kh,
                           stride_vb, stride_vr, stride_vk, stride_vh,
                           stride_bb, stride_br, stride_bq, stride_bk,
                           stride_mb, stride_mr, stride_mq,
                           stride_db, stride_dr, stride_dq,
                           DO, # FlashAttn Gradient Output
                           stride_dob, stride_dor, stride_doq, stride_doh,
                           DK,DV,DB, # Gradient Backpropogation
                           stride_dkb, stride_dkr, stride_dkk, stride_dkh,
                           stride_dvb, stride_dvr, stride_dvk, stride_dvh,
                           stride_dbb, stride_dbr, stride_dbq, stride_dbk,
                           seq_len_q, seq_len_kv,
                           SM_SCALE: tl.constexpr,
                           REL_SIZE: tl.constexpr,
                           KEY_SIZE: tl.constexpr,
                           BLOCK_Q: tl.constexpr,
                           BLOCK_K: tl.constexpr):
    start_k = tl.program_id(0)
    off_br = tl.program_id(1)
    off_b = (off_br // REL_SIZE).to(tl.int64)
    off_r = (off_br % REL_SIZE).to(tl.int64)
    q_offset = off_b * stride_qb + off_r * stride_qr
    k_offset = off_b * stride_kb + off_r * stride_kr
    v_offset = off_b * stride_vb + off_r * stride_vr
    b_offset = off_b * stride_bb + off_r * stride_br
    m_offset = off_b * stride_mb + off_r * stride_mr
    d_offset = off_b * stride_db + off_r * stride_dr

    do_offset = off_b * stride_dob + off_r * stride_dor

    dk_offset = off_b * stride_dkb + off_r * stride_dkr
    dv_offset = off_b * stride_dvb + off_r * stride_dvr
    db_offset = off_b * stride_dbb + off_r * stride_dbr

    QT_block_ptr = tl.make_block_ptr(
        base=Q+q_offset,
        shape=(KEY_SIZE, seq_len_q),
        strides=(stride_qh, stride_qq),
        offsets=(0, 0),
        block_shape=(KEY_SIZE, BLOCK_Q),
        order=(0, 1),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K+k_offset,
        shape=(seq_len_kv, KEY_SIZE),
        strides=(stride_kk, stride_kh),
        offsets=(start_k*BLOCK_K, 0),
        block_shape=(BLOCK_K, KEY_SIZE),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
            base=V+v_offset,
            shape=(seq_len_kv, KEY_SIZE),
            strides=(stride_vk, stride_vh),
            offsets=(start_k*BLOCK_K, 0),
            block_shape=(BLOCK_K, KEY_SIZE),
            order=(1, 0),
    )
    BT_block_ptr = tl.make_block_ptr(
        base=B+b_offset,
        shape=(seq_len_kv, seq_len_q),
        strides=(stride_bk, stride_bq),
        offsets=(start_k*BLOCK_K, 0),
        block_shape=(BLOCK_K, BLOCK_Q),
        order=(0, 1),
    )
    M_ptr = M + m_offset
    D_ptr = D + d_offset

    DOT_block_ptr = tl.make_block_ptr(
        base=DO+do_offset,
        shape=(KEY_SIZE, seq_len_q),
        strides=(stride_doh, stride_doq),
        offsets=(0, 0),
        block_shape=(KEY_SIZE, BLOCK_Q),
        order=(0, 1),
    )

    DK_block_ptr = tl.make_block_ptr(
        base=DK+dk_offset,
        shape=(seq_len_kv, KEY_SIZE),
        strides=(stride_dkk, stride_dkh),
        offsets=(start_k*BLOCK_K, 0),
        block_shape=(BLOCK_K, KEY_SIZE),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV+dv_offset,
        shape=(seq_len_kv, KEY_SIZE),
        strides=(stride_dvk, stride_dvh),
        offsets=(start_k*BLOCK_K, 0),
        block_shape=(BLOCK_K, KEY_SIZE),
        order=(1, 0),
    )
    DB_block_ptr = tl.make_block_ptr(
        base=DB+db_offset,
        shape=(seq_len_q, seq_len_kv),
        strides=(stride_dbq, stride_dbk),
        offsets=(0, start_k*BLOCK_K),
        block_shape=(BLOCK_Q, BLOCK_K),
        order=(1, 0),
    )

    k = tl.load(K_block_ptr,boundary_check=(0,),padding_option='zero')
    v = tl.load(V_block_ptr,boundary_check=(0,),padding_option='zero')

    dv = tl.zeros([BLOCK_K, KEY_SIZE], dtype=tl.float32)
    dk = tl.zeros([BLOCK_K, KEY_SIZE], dtype=tl.float32)

    for start_q in range(0, seq_len_q, BLOCK_Q):
        start_q = tl.multiple_of(start_q, BLOCK_Q)
        qT = tl.load(QT_block_ptr, boundary_check=(1,), padding_option='zero')
        bT = tl.load(BT_block_ptr, boundary_check=(0,1,), padding_option='zero')

        block_index_m = start_q + tl.arange(0, BLOCK_Q)
        mask = block_index_m<seq_len_q
        m_array_ptr = M_ptr + block_index_m*stride_mq
        m = tl.load(m_array_ptr, mask=mask)

        qkT = tl.dot(k, qT) * SM_SCALE + bT
        qkT = qkT * 1.44269504 # 1/log(2)
        pT = tl.math.exp2(qkT - m[None, :])
        doT = tl.load(DOT_block_ptr,boundary_check=(1,),padding_option='zero')
        # Compute dV.
        dv = tl.dot(pT, tl.trans(doT).to(tl.float32), acc=dv)

        # D (= delta)
        block_index_d = start_q + tl.arange(0, BLOCK_Q)
        mask = block_index_d<seq_len_q
        d_array_ptr = D_ptr + block_index_d*stride_dq
        Di = tl.load(d_array_ptr, mask=mask)

        # Compute dP and dS.
        dpT = tl.dot(v, doT)
        dsT = pT * (dpT - Di[None, :])
        ds = tl.trans(dsT)
        db = ds
        tl.store(DB_block_ptr, db.to(DB.dtype.element_ty), boundary_check=(0,1,))

        # Compute dK
        dsT = dsT * SM_SCALE
        dk = tl.dot(dsT, tl.trans(qT).to(tl.float32), acc=dk)
        # Increment pointers.
        QT_block_ptr = tl.advance(QT_block_ptr, (0, BLOCK_Q))
        BT_block_ptr = tl.advance(BT_block_ptr, (0, BLOCK_Q))
        DOT_block_ptr = tl.advance(DOT_block_ptr, (0, BLOCK_Q))
        DB_block_ptr = tl.advance(DB_block_ptr, (BLOCK_Q, 0))

    tl.store(DK_block_ptr, dk.to(DK.dtype.element_ty), boundary_check=(0,))
    tl.store(DV_block_ptr, dv.to(DV.dtype.element_ty), boundary_check=(0,))

@triton.autotune(list(filter(keep, configs)),
                 key=["seq_len_q","seq_len_kv"])
@triton.jit
def biased_flash_attn_dq(K, DB,
                         stride_kb, stride_kr, stride_kk, stride_kh,
                         stride_dbb, stride_dbr, stride_dbq, stride_dbk,
                         DQ,
                         stride_dqb, stride_dqr, stride_dqq, stride_dqh,
                         seq_len_q, seq_len_kv,
                         SM_SCALE: tl.constexpr,
                         REL_SIZE: tl.constexpr,
                         KEY_SIZE: tl.constexpr,
                         BLOCK_Q: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    start_q = tl.program_id(0)
    off_br = tl.program_id(1)
    off_b = (off_br // REL_SIZE).to(tl.int64)
    off_r = (off_br % REL_SIZE).to(tl.int64)
    k_offset = off_b * stride_kb + off_r * stride_kr
    db_offset = off_b * stride_dbb + off_r * stride_dbr
    dq_offset = off_b * stride_dqb + off_r * stride_dqr

    K_block_ptr = tl.make_block_ptr(
        base=K+k_offset,
        shape=(seq_len_kv, KEY_SIZE),
        strides=(stride_kk, stride_kh),
        offsets=(0, 0),
        block_shape=(BLOCK_K, KEY_SIZE),
        order=(1, 0),
    )
    DB_block_ptr = tl.make_block_ptr(
        base=DB+db_offset,
        shape=(seq_len_q, seq_len_kv),
        strides=(stride_dbq, stride_dbk),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_K),
        order=(1, 0),
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ+dq_offset,
        shape=(seq_len_q, KEY_SIZE),
        strides=(stride_dqq, stride_dqh),
        offsets=(start_q*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, KEY_SIZE),
        order=(1, 0),
    )

    dq = tl.zeros([BLOCK_Q, KEY_SIZE], dtype=tl.float32)
    for start_k in range(0, seq_len_kv, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        db = tl.load(DB_block_ptr, boundary_check=(0,1), padding_option='zero')
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option='zero')
        dq = tl.dot(db, k, acc=dq)
        DB_block_ptr = tl.advance(DB_block_ptr,(0, BLOCK_K))
        K_block_ptr = tl.advance(K_block_ptr,(BLOCK_K, 0))
    dq = dq*SM_SCALE
    tl.store(DQ_block_ptr, dq.to(DQ.dtype.element_ty), boundary_check=(0,))

def delta_calculation_unit_test(o: torch.Tensor, do: torch.Tensor):
    assert o.shape==do.shape
    batch_size, rel_size, seq_len_q, key_size = o.shape
    delta = torch.zeros(batch_size, rel_size, seq_len_q, device=o.device, dtype=o.dtype)
    grid = lambda meta: (triton.cdiv(seq_len_q, meta['BLOCK_Q']), batch_size*rel_size)
    delta_calculation[grid](o, do, delta,
                            o.stride(0),o.stride(1),o.stride(2),o.stride(3),
                            do.stride(0),do.stride(1),do.stride(2),do.stride(3),
                            delta.stride(0),delta.stride(1),delta.stride(2),
                            seq_len_q,
                            BLOCK_Q=512,
                            REL_SIZE=rel_size,KEY_SIZE=key_size)
    return delta

if __name__=='__main__':
    batch_size, rel_size, seq_len_q, seq_len_kv, key_size = 64, 256, 820, 820, 64
    o = torch.randn(batch_size,rel_size,seq_len_q,key_size,dtype=torch.float16,device='cuda')
    do = torch.randn(batch_size,rel_size,seq_len_q,key_size,dtype=torch.float16,device='cuda')

    delta = delta_calculation_unit_test(o,do)
    torch_output = (o*do).sum(-1)
    print((delta-torch_output).abs().max().item())
