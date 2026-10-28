import torch
import triton
import triton.language as tl
import math


def _attention_configs():
    return [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ]


@triton.jit
def _fwd_inner(
    X_, Q_, K_ptrs, V_ptrs,
    stride_t,
    start_m, offs_m, offs_n, offs_d,
    L, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, EVEN_L: tl.constexpr, EVEN_D: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        if not EVEN_L:
            hi = tl.minimum(hi, L)

    K_ptrs += lo * stride_t
    V_ptrs += lo * stride_t

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_D and (STAGE == 1 or EVEN_L):
            K_ = tl.load(K_ptrs)
        elif EVEN_D:
            K_ = tl.load(K_ptrs, mask=(start_n + offs_n[:, None]) < L, other=0.0)
        elif STAGE == 1 or EVEN_L:
            K_ = tl.load(K_ptrs, mask=offs_d[None, :] < D, other=0.0)
        else:
            K_ = tl.load(
                K_ptrs,
                mask=((start_n + offs_n[:, None]) < L) & (offs_d[None, :] < D),
                other=0.0,
            )

        W_ = tl.dot(Q_, tl.trans(K_))
        W_2 = W_ * W_
        W_4 = W_2 * W_2
        W_8 = W_4 * W_4
        A_ = (W_8 + 0.01) * W_
        if STAGE == 2:
            if EVEN_L:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            else:
                mask = (offs_m[:, None] >= (start_n + offs_n[None, :])) & (
                    (start_n + offs_n[None, :]) < L
                )
            A_ = tl.where(mask, A_, 0.0)

        if EVEN_D and (STAGE == 1 or EVEN_L):
            V_ = tl.load(V_ptrs)
        elif EVEN_D:
            V_ = tl.load(V_ptrs, mask=(start_n + offs_n[:, None]) < L, other=0.0)
        elif STAGE == 1 or EVEN_L:
            V_ = tl.load(V_ptrs, mask=offs_d[None, :] < D, other=0.0)
        else:
            V_ = tl.load(
                V_ptrs,
                mask=((start_n + offs_n[:, None]) < L) & (offs_d[None, :] < D),
                other=0.0,
            )

        X_ = tl.dot(A_.to(Q_.dtype), V_, acc=X_)
        K_ptrs += BLOCK_N * stride_t
        V_ptrs += BLOCK_N * stride_t

    return X_


@triton.autotune(
    configs=_attention_configs(),
    key=["L", "BLOCK_D"],
)
@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Y_ptr, Z_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_z_b, stride_z_h, stride_z_t,
    L,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_L: tl.constexpr,
    EVEN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = (tl.num_programs(0) - 1) - tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_b = pid_z // H
    pid_h = pid_z % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    bh_offset = pid_b * stride_b + pid_h * stride_h
    Q_offset = bh_offset + offs_m[:, None] * stride_t + offs_d[None, :] * stride_d
    if EVEN_L and EVEN_D:
        Q_ = tl.load(Q_ptr + Q_offset)
    elif EVEN_D:
        Q_ = tl.load(Q_ptr + Q_offset, mask=offs_m[:, None] < L, other=0.0)
    elif EVEN_L:
        Q_ = tl.load(Q_ptr + Q_offset, mask=offs_d[None, :] < D, other=0.0)
    else:
        Q_ = tl.load(
            Q_ptr + Q_offset,
            mask=(offs_m[:, None] < L) & (offs_d[None, :] < D),
            other=0.0,
        )
    Q_ = (Q_ * 2.0).to(Q_ptr.dtype.element_ty)

    K_ptrs = K_ptr + bh_offset + offs_n[:, None] * stride_t + offs_d[None, :] * stride_d
    V_ptrs = V_ptr + bh_offset + offs_n[:, None] * stride_t + offs_d[None, :] * stride_d

    X_ = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    X_ = _fwd_inner(
        X_, Q_, K_ptrs, V_ptrs,
        stride_t,
        start_m, offs_m, offs_n, offs_d,
        L, D,
        BLOCK_M, BLOCK_N,
        1, EVEN_L, EVEN_D,
    )
    X_ = _fwd_inner(
        X_, Q_, K_ptrs, V_ptrs,
        stride_t,
        start_m, offs_m, offs_n, offs_d,
        L, D,
        BLOCK_M, BLOCK_N,
        2, EVEN_L, EVEN_D,
    )

    Z_sq = tl.sum(X_ * X_, axis=1) * (1.0 / D) + 1e-12
    Z_ = tl.rsqrt(Z_sq)
    Y_ = (X_ * Z_[:, None]).to(Q_ptr.dtype.element_ty)

    Z_offset = pid_b * stride_z_b + pid_h * stride_z_h + offs_m * stride_z_t
    if EVEN_L and EVEN_D:
        tl.store(Y_ptr + Q_offset, Y_)
        tl.store(Z_ptr + Z_offset, Z_)
    elif EVEN_D:
        q_mask = offs_m < L
        tl.store(Y_ptr + Q_offset, Y_, mask=q_mask[:, None])
        tl.store(Z_ptr + Z_offset, Z_, mask=q_mask)
    elif EVEN_L:
        tl.store(Y_ptr + Q_offset, Y_, mask=offs_d[None, :] < D)
        tl.store(Z_ptr + Z_offset, Z_)
    else:
        q_mask = offs_m < L
        tl.store(Y_ptr + Q_offset, Y_, mask=q_mask[:, None] & (offs_d[None, :] < D))
        tl.store(Z_ptr + Z_offset, Z_, mask=q_mask)


@triton.jit
def _bwd_dq_inner(
    dQ_, Q_, dX_, K_ptrs, V_ptrs,
    stride_t,
    start_m, offs_m, offs_n, offs_d,
    L, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, EVEN_L: tl.constexpr, EVEN_D: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        if not EVEN_L:
            hi = tl.minimum(hi, L)

    K_ptrs += lo * stride_t
    V_ptrs += lo * stride_t

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_D and (STAGE == 1 or EVEN_L):
            K_ = tl.load(K_ptrs)
        elif EVEN_D:
            K_ = tl.load(K_ptrs, mask=(start_n + offs_n[:, None]) < L, other=0.0)
        elif STAGE == 1 or EVEN_L:
            K_ = tl.load(K_ptrs, mask=offs_d[None, :] < D, other=0.0)
        else:
            K_ = tl.load(
                K_ptrs,
                mask=((start_n + offs_n[:, None]) < L) & (offs_d[None, :] < D),
                other=0.0,
            )

        W_ = tl.dot(Q_, tl.trans(K_))
        W_2 = W_ * W_
        W_4 = W_2 * W_2
        W_8 = W_4 * W_4
        F_ = 9.0 * W_8 + 0.01
        if STAGE == 2:
            if EVEN_L:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            else:
                mask = (offs_m[:, None] >= (start_n + offs_n[None, :])) & (
                    (start_n + offs_n[None, :]) < L
                )
            F_ = tl.where(mask, F_, 0.0)

        if EVEN_D and (STAGE == 1 or EVEN_L):
            V_ = tl.load(V_ptrs)
        elif EVEN_D:
            V_ = tl.load(V_ptrs, mask=(start_n + offs_n[:, None]) < L, other=0.0)
        elif STAGE == 1 or EVEN_L:
            V_ = tl.load(V_ptrs, mask=offs_d[None, :] < D, other=0.0)
        else:
            V_ = tl.load(
                V_ptrs,
                mask=((start_n + offs_n[:, None]) < L) & (offs_d[None, :] < D),
                other=0.0,
            )

        tmp = tl.dot(dX_, tl.trans(V_)) * F_
        dQ_ = tl.dot(tmp.to(Q_.dtype), K_, acc=dQ_)
        K_ptrs += BLOCK_N * stride_t
        V_ptrs += BLOCK_N * stride_t

    return dQ_


@triton.autotune(
    configs=_attention_configs(),
    key=["L", "BLOCK_D"],
)
@triton.jit
def _bwd_kernel_dQ(
    Q_ptr, K_ptr, V_ptr, Y_ptr, Z_ptr,
    dY_ptr, dX_ptr, dQ_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_z_b, stride_z_h, stride_z_t,
    L,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_L: tl.constexpr,
    EVEN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = (tl.num_programs(0) - 1) - tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_b = pid_z // H
    pid_h = pid_z % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    bh_offset = pid_b * stride_b + pid_h * stride_h
    Q_offset = bh_offset + offs_m[:, None] * stride_t + offs_d[None, :] * stride_d
    Z_offset = pid_b * stride_z_b + pid_h * stride_z_h + offs_m * stride_z_t

    if EVEN_L and EVEN_D:
        Q_ = tl.load(Q_ptr + Q_offset)
        dY_ = tl.load(dY_ptr + Q_offset).to(tl.float32)
        Y_ = tl.load(Y_ptr + Q_offset).to(tl.float32)
        Z_ = tl.load(Z_ptr + Z_offset)
    elif EVEN_D:
        q_mask = offs_m < L
        Q_ = tl.load(Q_ptr + Q_offset, mask=q_mask[:, None], other=0.0)
        dY_ = tl.load(dY_ptr + Q_offset, mask=q_mask[:, None], other=0.0).to(tl.float32)
        Y_ = tl.load(Y_ptr + Q_offset, mask=q_mask[:, None], other=0.0).to(tl.float32)
        Z_ = tl.load(Z_ptr + Z_offset, mask=q_mask, other=0.0)
    elif EVEN_L:
        d_mask = offs_d[None, :] < D
        Q_ = tl.load(Q_ptr + Q_offset, mask=d_mask, other=0.0)
        dY_ = tl.load(dY_ptr + Q_offset, mask=d_mask, other=0.0).to(tl.float32)
        Y_ = tl.load(Y_ptr + Q_offset, mask=d_mask, other=0.0).to(tl.float32)
        Z_ = tl.load(Z_ptr + Z_offset)
    else:
        q_mask = offs_m < L
        qd_mask = q_mask[:, None] & (offs_d[None, :] < D)
        Q_ = tl.load(Q_ptr + Q_offset, mask=qd_mask, other=0.0)
        dY_ = tl.load(dY_ptr + Q_offset, mask=qd_mask, other=0.0).to(tl.float32)
        Y_ = tl.load(Y_ptr + Q_offset, mask=qd_mask, other=0.0).to(tl.float32)
        Z_ = tl.load(Z_ptr + Z_offset, mask=q_mask, other=0.0)

    Q_ = (Q_ * 2.0).to(Q_ptr.dtype.element_ty)
    dYZ_ = dY_ * Z_[:, None]
    dYZ2_ = tl.sum(dYZ_ * Y_, axis=1, keep_dims=True) * (1.0 / D)
    dX_ = (dYZ_ - dYZ2_ * Y_).to(Q_ptr.dtype.element_ty)

    if EVEN_L and EVEN_D:
        tl.store(dX_ptr + Q_offset, dX_)
    elif EVEN_D:
        tl.store(dX_ptr + Q_offset, dX_, mask=offs_m[:, None] < L)
    elif EVEN_L:
        tl.store(dX_ptr + Q_offset, dX_, mask=offs_d[None, :] < D)
    else:
        tl.store(dX_ptr + Q_offset, dX_, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D))

    K_ptrs = K_ptr + bh_offset + offs_n[:, None] * stride_t + offs_d[None, :] * stride_d
    V_ptrs = V_ptr + bh_offset + offs_n[:, None] * stride_t + offs_d[None, :] * stride_d

    dQ_ = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    dQ_ = _bwd_dq_inner(
        dQ_, Q_, dX_, K_ptrs, V_ptrs,
        stride_t,
        start_m, offs_m, offs_n, offs_d,
        L, D,
        BLOCK_M, BLOCK_N,
        1, EVEN_L, EVEN_D,
    )
    dQ_ = _bwd_dq_inner(
        dQ_, Q_, dX_, K_ptrs, V_ptrs,
        stride_t,
        start_m, offs_m, offs_n, offs_d,
        L, D,
        BLOCK_M, BLOCK_N,
        2, EVEN_L, EVEN_D,
    )

    dQ_ = (dQ_ * 2.0).to(Q_ptr.dtype.element_ty)
    if EVEN_L and EVEN_D:
        tl.store(dQ_ptr + Q_offset, dQ_)
    elif EVEN_D:
        tl.store(dQ_ptr + Q_offset, dQ_, mask=offs_m[:, None] < L)
    elif EVEN_L:
        tl.store(dQ_ptr + Q_offset, dQ_, mask=offs_d[None, :] < D)
    else:
        tl.store(dQ_ptr + Q_offset, dQ_, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D))


@triton.jit
def _bwd_dkdv_inner(
    dK_, dV_, K_, V_, Q_ptrs, dX_ptrs,
    stride_t,
    start_n, offs_n, offs_m, offs_d,
    L, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, EVEN_L: tl.constexpr, EVEN_D: tl.constexpr,
):
    if STAGE == 2:
        lo, hi = start_n * BLOCK_M, (start_n + 1) * BLOCK_M
        if not EVEN_L:
            hi = tl.minimum(hi, L)
    else:
        lo, hi = (start_n + 1) * BLOCK_M, L

    Q_ptrs += lo * stride_t
    dX_ptrs += lo * stride_t

    for start_m in range(lo, hi, BLOCK_N):
        start_m = tl.multiple_of(start_m, BLOCK_N)
        if EVEN_L and EVEN_D:
            Q_ = tl.load(Q_ptrs)
            dX_ = tl.load(dX_ptrs)
        elif EVEN_D:
            q_mask = (start_m + offs_m[:, None]) < L
            Q_ = tl.load(Q_ptrs, mask=q_mask, other=0.0)
            dX_ = tl.load(dX_ptrs, mask=q_mask, other=0.0)
        elif EVEN_L:
            d_mask = offs_d[None, :] < D
            Q_ = tl.load(Q_ptrs, mask=d_mask, other=0.0)
            dX_ = tl.load(dX_ptrs, mask=d_mask, other=0.0)
        else:
            qd_mask = ((start_m + offs_m[:, None]) < L) & (offs_d[None, :] < D)
            Q_ = tl.load(Q_ptrs, mask=qd_mask, other=0.0)
            dX_ = tl.load(dX_ptrs, mask=qd_mask, other=0.0)

        W_ = tl.dot(K_, tl.trans(Q_))
        W_2 = W_ * W_
        W_4 = W_2 * W_2
        W_8 = W_4 * W_4
        A_ = (W_8 + 0.01) * W_
        if STAGE == 2:
            if EVEN_L:
                mask = offs_n[:, None] <= (start_m + offs_m[None, :])
            else:
                mask = (offs_n[:, None] <= (start_m + offs_m[None, :])) & (
                    (start_m + offs_m[None, :]) < L
                )
            A_ = tl.where(mask, A_, 0.0)
            F_ = tl.where(mask, 9.0 * W_8 + 0.01, 0.0)
        else:
            F_ = 9.0 * W_8 + 0.01

        dV_ = tl.dot(A_.to(K_.dtype), dX_, acc=dV_)
        tmp = tl.dot(V_, tl.trans(dX_)) * F_
        dK_ = tl.dot(tmp.to(K_.dtype), Q_, acc=dK_)
        Q_ptrs += BLOCK_N * stride_t
        dX_ptrs += BLOCK_N * stride_t

    return dK_, dV_


@triton.autotune(
    configs=_attention_configs(),
    key=["L", "BLOCK_D"],
)
@triton.jit
def _bwd_kernel_dKdV(
    Q_ptr, K_ptr, V_ptr,
    dX_ptr, dK_ptr, dV_ptr,
    stride_b, stride_h, stride_t, stride_d,
    L,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_L: tl.constexpr,
    EVEN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_b = pid_z // H
    pid_h = pid_z % H

    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    bh_offset = pid_b * stride_b + pid_h * stride_h
    KV_offset = bh_offset + offs_n[:, None] * stride_t + offs_d[None, :] * stride_d
    if EVEN_L and EVEN_D:
        K_ = tl.load(K_ptr + KV_offset)
        V_ = tl.load(V_ptr + KV_offset)
    elif EVEN_D:
        kv_mask = offs_n[:, None] < L
        K_ = tl.load(K_ptr + KV_offset, mask=kv_mask, other=0.0)
        V_ = tl.load(V_ptr + KV_offset, mask=kv_mask, other=0.0)
    elif EVEN_L:
        d_mask = offs_d[None, :] < D
        K_ = tl.load(K_ptr + KV_offset, mask=d_mask, other=0.0)
        V_ = tl.load(V_ptr + KV_offset, mask=d_mask, other=0.0)
    else:
        kvd_mask = (offs_n[:, None] < L) & (offs_d[None, :] < D)
        K_ = tl.load(K_ptr + KV_offset, mask=kvd_mask, other=0.0)
        V_ = tl.load(V_ptr + KV_offset, mask=kvd_mask, other=0.0)

    K_ = (K_ * 2.0).to(Q_ptr.dtype.element_ty)
    dK_ = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    dV_ = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    Q_ptrs = Q_ptr + bh_offset + offs_m[:, None] * stride_t + offs_d[None, :] * stride_d
    dX_ptrs = dX_ptr + bh_offset + offs_m[:, None] * stride_t + offs_d[None, :] * stride_d

    dK_, dV_ = _bwd_dkdv_inner(
        dK_, dV_, K_, V_, Q_ptrs, dX_ptrs,
        stride_t,
        start_n, offs_n, offs_m, offs_d,
        L, D,
        BLOCK_M, BLOCK_N,
        2, EVEN_L, EVEN_D,
    )
    dK_, dV_ = _bwd_dkdv_inner(
        dK_, dV_, K_, V_, Q_ptrs, dX_ptrs,
        stride_t,
        start_n, offs_n, offs_m, offs_d,
        L, D,
        BLOCK_M, BLOCK_N,
        1, EVEN_L, EVEN_D,
    )

    dK_ = (dK_ * 2.0).to(Q_ptr.dtype.element_ty)
    dV_ = dV_.to(Q_ptr.dtype.element_ty)

    if EVEN_L and EVEN_D:
        tl.store(dK_ptr + KV_offset, dK_)
        tl.store(dV_ptr + KV_offset, dV_)
    elif EVEN_D:
        kv_mask = offs_n[:, None] < L
        tl.store(dK_ptr + KV_offset, dK_, mask=kv_mask)
        tl.store(dV_ptr + KV_offset, dV_, mask=kv_mask)
    elif EVEN_L:
        d_mask = offs_d[None, :] < D
        tl.store(dK_ptr + KV_offset, dK_, mask=d_mask)
        tl.store(dV_ptr + KV_offset, dV_, mask=d_mask)
    else:
        kvd_mask = (offs_n[:, None] < L) & (offs_d[None, :] < D)
        tl.store(dK_ptr + KV_offset, dK_, mask=kvd_mask)
        tl.store(dV_ptr + KV_offset, dV_, mask=kvd_mask)


class PolyAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q_: torch.Tensor,
        K_: torch.Tensor,
        V_: torch.Tensor,
    ):
        B, L, H, D = Q_.shape
        Z = B * H
        BLOCK_D = max(16, triton.next_power_of_2(D))
        EVEN_L = (L % 128) == 0
        EVEN_D = D == BLOCK_D

        Q_ = Q_.contiguous()
        K_ = K_.contiguous()
        V_ = V_.contiguous()
        Y_ = torch.empty_like(Q_)
        Z_ = torch.empty((B, H, L), device=Q_.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]), Z)
        _fwd_kernel[grid](
            Q_, K_, V_, Y_, Z_,
            Y_.stride(0), Y_.stride(2), Y_.stride(1), Y_.stride(3),
            Z_.stride(0), Z_.stride(1), Z_.stride(2),
            L,
            H=H,
            D=D,
            BLOCK_D=BLOCK_D,
            EVEN_L=EVEN_L,
            EVEN_D=EVEN_D,
        )

        ctx.save_for_backward(Q_, K_, V_, Y_, Z_)
        ctx.shape = (B, L, H, D, Z, BLOCK_D, EVEN_L, EVEN_D)
        return Y_

    @staticmethod
    def backward(ctx, dY_):
        Q_, K_, V_, Y_, Z_ = ctx.saved_tensors
        B, L, H, D, Z, BLOCK_D, EVEN_L, EVEN_D = ctx.shape

        dY_ = dY_.contiguous()
        dX_ = torch.empty_like(Q_)
        dQ_ = torch.empty_like(Q_)
        dK_ = torch.empty_like(K_)
        dV_ = torch.empty_like(V_)

        grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]), Z)
        _bwd_kernel_dQ[grid](
            Q_, K_, V_, Y_, Z_,
            dY_, dX_, dQ_,
            V_.stride(0), V_.stride(2), V_.stride(1), V_.stride(3),
            Z_.stride(0), Z_.stride(1), Z_.stride(2),
            L,
            H=H,
            D=D,
            BLOCK_D=BLOCK_D,
            EVEN_L=EVEN_L,
            EVEN_D=EVEN_D,
        )

        _bwd_kernel_dKdV[grid](
            Q_, K_, V_,
            dX_, dK_, dV_,
            V_.stride(0), V_.stride(2), V_.stride(1), V_.stride(3),
            L,
            H=H,
            D=D,
            BLOCK_D=BLOCK_D,
            EVEN_L=EVEN_L,
            EVEN_D=EVEN_D,
        )

        return dQ_, dK_, dV_


def poly_attention_naive(q_, k_, v_):
    q_ = q_.transpose(1, 2)
    k_ = k_.transpose(1, 2)
    v_ = v_.transpose(1, 2)
    W_ = q_ @ k_.transpose(2, 3)
    A_ = W_.pow(9) + 0.01 * W_
    A_ = torch.tril(A_)
    Y_ = A_ @ v_
    Y_ = torch.nn.functional.rms_norm(Y_, (Y_.shape[-1],))
    Y_ = Y_.transpose(1, 2)
    return Y_


def poly_attention(q_, k_, v_):
    return PolyAttention.apply(q_, k_, v_)