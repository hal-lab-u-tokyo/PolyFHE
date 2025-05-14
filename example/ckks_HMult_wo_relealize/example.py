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
res_axax = pf.end(mult_axax, 0, 0)
res_axbx = pf.end(add_axbx, 0, prm.N * prm.L)
res_bxbx = pf.end(mult_bxbx, 0, prm.N * prm.L * 2)
target.append(res_axax)
target.append(res_axbx)
target.append(res_bxbx)

# Compile
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile(target, filepath)

print("target:", target)
