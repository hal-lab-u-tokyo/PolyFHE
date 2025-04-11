from polyfhe import PolyFHE

pf = PolyFHE()

a = pf.add(1, 2)
b = pf.sub(a, 3)
c = pf.mul(b, 4)

pf.compile(c)