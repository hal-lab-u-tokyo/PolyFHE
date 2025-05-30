from pypolyfhe import PolyFHE, Params, PrecomputedValue
import os

pf = PolyFHE()
# prm = Params(N=2**15, L=15, dnum=5)
prm = Params(N=2**16, L=30, dnum=5)
# prm = Params(N=2**16, L=35, dnum=7) 
# prm = Params(N=2**14, L=6, dnum=3)
# prm = Params(N=2**16, L=40, dnum=20)
print(prm)

target = []

"""
# HAdd
e_ct0_ax = pf.init("ct0_ax", idx_ct=0, offset=0)
e_ct0_bx = pf.init("ct0_bx", idx_ct=0, offset=prm.N * prm.L)
e_ct1_ax = pf.init("ct1_ax", idx_ct=1, offset=0)
e_ct1_bx = pf.init("ct1_bx", idx_ct=1, offset=prm.N * prm.L)
add_axax = pf.add(e_ct0_ax, e_ct1_ax, "MultAxAx", start_limb=0, end_limb=prm.L)
add_bxbx = pf.add(e_ct0_bx, e_ct1_bx, "MultBxBx", start_limb=0, end_limb=prm.L)
res_axax = pf.end(add_axax, 0, 0)
res_bxbx = pf.end(add_bxbx, 0, prm.N * prm.L * 1)

target.append(res_axax)
target.append(res_bxbx)

current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile(target, filepath)

print("target:", target)
"""






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
scale_for_bconv = pf.mul_const(inttp1, "ScaleForBConv", PrecomputedValue.ModUp, 0, prm.L)
accum_list = []
for beta_idx in range(prm.get_beta(prm.L - 1)):
    print("beta_idx:", beta_idx)
    bconv = pf.bconv(scale_for_bconv, f"BConv{beta_idx}", prm.L, beta_idx, prm.alpha)
    res_bconv = pf.end(bconv, 1, prm.N * (prm.L + prm.K) * beta_idx)
    target.append(res_bconv)
    """
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
    """

"""
accum = pf.mul_key_accum(accum_list, "MultKeyAccum", start_limb=0, end_limb=prm.L + prm.K, beta=prm.get_beta(prm.L - 1))
inttp2_ax = pf.ntt(
    accum,
    "INTT_Ax",
    if_forward=False,
    if_phase1=False,
    start_limb=prm.L,
    end_limb=prm.L + prm.K,
    out_start_limb=0,
    out_end_limb=prm.L + prm.K,
)
inttp1_ax = pf.ntt(
    inttp2_ax,
    "INTT1_Ax",
    if_forward=False,
    if_phase1=True,
    start_limb=prm.L,
    end_limb=prm.L + prm.K,
    out_start_limb=0,
    out_end_limb=prm.L + prm.K,
)
inttp2_bx = pf.ntt(
    accum,
    "INTT_Bx",
    if_forward=False,
    if_phase1=False,
    start_limb=prm.L,
    end_limb=prm.L + prm.K,
    out_start_limb=0,
    out_end_limb=prm.L + prm.K,
)
inttp1_bx = pf.ntt(
    inttp2_bx,
    "INTT1_Bx",
    if_forward=False,
    if_phase1=True,
    start_limb=prm.L,
    end_limb=prm.L + prm.K,
    out_start_limb=0,
    out_end_limb=prm.L + prm.K,
)
moddown_scale_ax = pf.mul_const(inttp1_ax, "ModDownScaleAx", PrecomputedValue.ModDown, prm.L, prm.L + prm.K)
moddown_scale_bx = pf.mul_const(inttp1_bx, "ModDownScaleBx", PrecomputedValue.ModDown, prm.L, prm.L + prm.K)
bconv_ax = pf.bconv_general(moddown_scale_ax, f"BConv_ax", prm.L, prm.L + prm.K, 0, prm.L)
bconv_bx = pf.bconv_general(moddown_scale_bx, f"BConv_bx", prm.L, prm.L + prm.K, 0, prm.L)
nttp1_ax = pf.ntt(
        bconv_ax,
        f"NTTP1_ax",
        if_forward=True,
        if_phase1=True,
        start_limb=0,
        end_limb=prm.L,
) 
nttp2_ax = pf.ntt(
        nttp1_ax,
        f"NTTP2_ax",
        if_forward=True,
        if_phase1=False,
        start_limb=0,
        end_limb=prm.L,
)
nttp1_bx = pf.ntt(
        bconv_bx,
        f"NTTP1_bx",
        if_forward=True,
        if_phase1=True,
        start_limb=0,
        end_limb=prm.L,
)
nttp2_bx = pf.ntt(
        nttp1_bx,
        f"NTTP2_bx",
        if_forward=True,
        if_phase1=False,
        start_limb=0,
        end_limb=prm.L,
)
"""
# res_ax = pf.end(nttp2_ax, 1, 0)
# res_bx = pf.end(nttp2_bx, 1, prm.N * (prm.L + prm.K))
# sub_ax = pf.add(accum, nttp2_ax, "SubAx", start_limb=0, end_limb=prm.L)
# sub_bx = pf.add(accum, nttp2_bx, "SubBx", start_limb=0, end_limb=prm.L)
# res_ax = pf.end(sub_ax, 1, 0)
# res_bx = pf.end(sub_bx, 1, prm.N * (prm.L + prm.K))

# add_axax = pf.add(mult_axax, nttp2_ax, "AddAxAx", start_limb=0, end_limb=prm.L)
# add_axbx = pf.add(mult_axbx, nttp2_bx, "AddAxBx", start_limb=0, end_limb=prm.L)
# res_axax = pf.end(add_axax, 0, 0)
# res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)

# target.append(res_ax)
# target.append(res_bx)

res_axax = pf.end(mult_axax, 0, 0)
res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)
res_bxbx = pf.end(mult_bxbx, 0, prm.N * prm.L * 2)
# res_ks = pf.end(scale_for_bconv, 1, 0)

target.append(res_axax)
target.append(res_axbx)
target.append(res_bxbx)
# target.append(res_ks)

# Compile
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile(target, filepath)

print("target:", target)
