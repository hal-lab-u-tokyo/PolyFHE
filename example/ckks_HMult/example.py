from pypolyfhe import PolyFHE, Params
import os

pf = PolyFHE()
prm = Params(N=2 ** 15, L=18, dnum=9)
print(prm)

# HMult
e_ct0_ax = pf.init(0, 0, "ct0_ax")
e_ct0_bx = pf.init(0, prm.N * prm.L, "ct0_bx")
e_ct1_ax = pf.init(1, 0, "ct1_ax")
e_ct1_bx = pf.init(1, prm.N * prm.L, "ct1_bx")
mult_axax = pf.mul(e_ct0_ax, e_ct1_ax, "MultAxAx", prm.L, 0, prm.L)
mult_axbx = pf.mul(e_ct0_ax, e_ct1_bx, "MultAxBx", prm.L, 0, prm.L)
mult_bxax = pf.mul(e_ct0_bx, e_ct1_ax, "MultBxAx", prm.L, 0, prm.L)
add_axbx = pf.add(mult_axbx, mult_bxax, "AddAxBx", prm.L, 0, prm.L)
mult_bxbx = pf.mul(e_ct0_bx, e_ct1_bx, "MultBxBx", prm.L, 0, prm.L)
inttp2 = pf.intt_phase2(mult_bxbx, "iNTTP2", prm.L, 0, prm.L)
inttp1 = pf.intt_phase1(inttp2, "iNTT1", prm.L, 0, prm.L)
decomp = pf.decomp(inttp1, "Decomp", prm.L, 0, prm.L, prm.alpha, prm.get_beta(prm.L - 1))
res_axax = pf.end(mult_axax, 0, 0)
res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)
res_bxbx = pf.end(decomp, 0, prm.N * prm.L * 2)

# Compile
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile([res_axax, res_axbx, res_bxbx], filepath)