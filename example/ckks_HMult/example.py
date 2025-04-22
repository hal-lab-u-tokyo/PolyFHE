from pypolyfhe import PolyFHE, Params
import os

pf = PolyFHE()
prm = Params(2 ** 15, 18)

e_ct0_ax = pf.init(0, 0, "ct0_ax")
e_ct0_bx = pf.init(0, prm.N * prm.L, "ct0_bx")
e_ct1_ax = pf.init(1, 0, "ct1_ax")
e_ct1_bx = pf.init(1, prm.N * prm.L, "ct1_bx")
mult_axax = pf.mul(e_ct0_ax, e_ct1_ax, "mult_axax", prm.L, 0, prm.L)
mult_axbx = pf.mul(e_ct0_ax, e_ct1_bx, "mult_axbx", prm.L, 0, prm.L)
mult_bxax = pf.mul(e_ct0_bx, e_ct1_ax, "mult_bxax", prm.L, 0, prm.L)
add_axbx = pf.add(mult_axbx, mult_bxax, "add_axbx", prm.L, 0, prm.L)
mult_bxbx = pf.mul(e_ct0_bx, e_ct1_bx, "mult_bxbx", prm.L, 0, prm.L)
inttp2 = pf.intt_phase2(mult_bxbx, "inttp2", prm.L, 0, prm.L)
inttp1 = pf.intt_phase1(inttp2, "inttp1", prm.L, 0, prm.L)
decomp = pf.decomp(inttp1, "decomp", prm.L, 0, prm.L)
res_axax = pf.end(mult_axax, 0, 0)
res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)
res_bxbx = pf.end(decomp, 0, prm.N * prm.L * 2)
"""
ct0_bx = pf.init_op(pf.ct0, params.N * params.L)
ct1_ax = pf.init_op(pf.ct1, 0)
ct1_bx = pf.init_op(pf.ct1, params.N * params.L)
res = pf.mul(ct0_ax, ct1_ax)
"""
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile([res_axax, res_axbx, res_bxbx], filepath)