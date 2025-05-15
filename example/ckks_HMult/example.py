from pypolyfhe import PolyFHE, Params
import os

pf = PolyFHE()
# prm = Params(N=2**15, L=18, dnum=9)
prm = Params(N=2**16, L=30, dnum=5)
print(prm)

target = []

# HMult
e_ct0_ax = pf.init("ct0_ax", idx_ct=0, offset=0)
e_ct0_bx = pf.init("ct0_bx", idx_ct=0, offset=prm.N * prm.L)
e_ct1_ax = pf.init("ct1_ax", idx_ct=1, offset=0)
e_ct1_bx = pf.init("ct1_bx", idx_ct=1, offset=prm.N * prm.L)
mult_axax = pf.mul(e_ct0_ax, e_ct1_ax, "MultAxAx", start_limb=0, end_limb=prm.L)
mult_axbx = pf.mul(e_ct0_ax, e_ct1_bx, "MultAxBx", start_limb=0, end_limb=prm.L)
mult_bxax = pf.mul(e_ct0_bx, e_ct1_ax, "MultBxAx", start_limb=0, end_limb=prm.L)
add_axbx = pf.add(mult_axbx, mult_bxax, "AddAxBx", start_limb=0, end_limb=prm.L)
mult_bxbx = pf.mul(e_ct0_bx, e_ct1_bx, "MultBxBx", start_limb=0, end_limb=prm.L)
inttp2 = pf.ntt(
    mult_bxbx,
    "iNTTP2",
    if_forward=False,
    if_phase1=False,
    start_limb=0,
    end_limb=prm.L,
    exclude_start=0,
    exclude_end=0,
)
inttp1 = pf.ntt(
    inttp2,
    "iNTTP1",
    if_forward=False,
    if_phase1=True,
    start_limb=0,
    end_limb=prm.L,
    exclude_start=0,
    exclude_end=0,
)
scale_for_bconv = pf.mul_const(inttp1, "ScaleForBConv", 0, prm.L)
accum_list = []
for beta_idx in range(prm.get_beta(prm.L - 1)):
    print("beta_idx:", beta_idx)
    bconv = pf.bconv(scale_for_bconv, f"BConv{beta_idx}", prm.L, beta_idx, prm.alpha)
    nttp1_after_bconv = pf.ntt(
        bconv,
        f"NTTP1{beta_idx}",
        if_forward=True,
        if_phase1=True,
        start_limb=0,
        end_limb=prm.L + prm.K,
        exclude_start=prm.alpha * beta_idx,
        exclude_end=prm.alpha * (beta_idx + 1),
    )
    nttp2_after_bconv = pf.ntt(
        nttp1_after_bconv,
        f"NTTP2{beta_idx}",
        if_forward=True,
        if_phase1=False,
        start_limb=0,
        end_limb=prm.L + prm.K,
        exclude_start=prm.alpha * beta_idx,
        exclude_end=prm.alpha * (beta_idx + 1),
    )
    accum_list.append(nttp2_after_bconv)

accum = pf.mul_key_accum(accum_list, "MultKeyAccum", start_limb=0, end_limb=prm.L + prm.K, beta=prm.get_beta(prm.L - 1))
inttp2_ax = pf.ntt(
    accum,
    "INTT_Ax",
    if_forward=False,
    if_phase1=False,
    start_limb=0,
    end_limb=prm.L + prm.K,
    exclude_start=0,
    exclude_end=0,
)
inttp1_ax = pf.ntt(
    inttp2_ax,
    "INTT1_Ax",
    if_forward=False,
    if_phase1=True,
    start_limb=0,
    end_limb=prm.L + prm.K,
    exclude_start=0,
    exclude_end=0,
)
inttp2_bx = pf.ntt(
    accum,
    "INTT_Bx",
    if_forward=False,
    if_phase1=False,
    start_limb=0,
    end_limb=prm.L + prm.K,
    exclude_start=0,
    exclude_end=0,
)
inttp1_bx = pf.ntt(
    inttp2_bx,
    "INTT1_Bx",
    if_forward=False,
    if_phase1=True,
    start_limb=0,
    end_limb=prm.L + prm.K,
    exclude_start=0,
    exclude_end=0,
)
res_ax = pf.end(inttp1_ax, 1, 0)
res_bx = pf.end(inttp1_bx, 1, prm.N * (prm.L + prm.K))
"""
res_ax = pf.end(accum, 1, 0)
res_bx = pf.end(accum, 1, prm.N * (prm.L + prm.K))
"""
res_axax = pf.end(mult_axax, 0, 0)
res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)
res_bxbx = pf.end(mult_bxbx, 0, prm.N * prm.L * 2)

target.append(res_axax)
target.append(res_axbx)
target.append(res_bxbx)
target.append(res_ax)
target.append(res_bx)

# Compile
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile(target, filepath)

print("target:", target)
