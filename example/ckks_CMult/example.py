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
e_const = pf.init("plain_const", idx_ct=1, offset=0)
mult_ax = pf.mul(e_ct0_ax, e_const, "MultAxAx", start_limb=0, end_limb=prm.L)
mult_bx = pf.mul(e_ct0_bx, e_const, "MultAxBx", start_limb=0, end_limb=prm.L)
res_ax = pf.end(mult_ax, 0, 0)
res_bx = pf.end(mult_bx, 0, prm.N * prm.L)
target.append(res_ax)
target.append(res_bx)

# Compile
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "output")
pf.compile(target, filepath)

print("target:", target)
