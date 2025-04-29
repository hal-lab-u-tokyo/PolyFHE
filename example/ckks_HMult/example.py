from pypolyfhe import PolyFHE, Params
import os

pf = PolyFHE()
prm = Params(N=2**15, L=18, dnum=9)
print(prm)

target = []

# HMult
e_ct0_ax = pf.init(0, 0, "ct0_ax")
e_ct0_bx = pf.init(0, prm.N * prm.L, "ct0_bx")
e_ct1_ax = pf.init(1, 0, "ct1_ax")
e_ct1_bx = pf.init(1, prm.N * prm.L, "ct1_bx")
mult_axax = pf.mul(e_ct0_ax, e_ct1_ax, "MultAxAx", 0, prm.L)
mult_axbx = pf.mul(e_ct0_ax, e_ct1_bx, "MultAxBx", 0, prm.L)
mult_bxax = pf.mul(e_ct0_bx, e_ct1_ax, "MultBxAx", 0, prm.L)
add_axbx = pf.add(mult_axbx, mult_bxax, "AddAxBx", 0, prm.L)
mult_bxbx = pf.mul(e_ct0_bx, e_ct1_bx, "MultBxBx", 0, prm.L)
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
for beta_idx in range(prm.get_beta(prm.L - 1)):
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
    # nttp2_after_bconv = pf.ntt(
    #     nttp1_after_bconv,
    #     f"NTTP2{beta_idx}",
    #     if_forward=True,
    #     if_phase1=False,
    #     start_limb=0,
    #     end_limb=prm.L + prm.K,
    #     exclude_start=prm.alpha * beta_idx,
    #     exclude_end=prm.alpha * (beta_idx + 1),
    # )
    res = pf.end(nttp1_after_bconv, 1, prm.N * (prm.L + prm.K) * beta_idx)
    target.append(res)
res_axax = pf.end(mult_axax, 0, 0)
res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)
res_bxbx = pf.end(mult_bxbx, 0, prm.N * prm.L * 2)
# res_intt = pf.end(scale_for_bconv, 1, 0)
target.append(res_axax)
target.append(res_axbx)
target.append(res_bxbx)
# target.append(res_intt)

# Compile
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile(target, filepath)

print("target:", target)
