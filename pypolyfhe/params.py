import math

class Params:
    """
    Class to hold the parameters for the encryption scheme.
    """

    def __init__(self, N: int, L: int, dnum: int):
        self.N = N  # Degree of the polynomial
        self.L = L  # Number of max levels
        self.dnum = dnum # Number of Decompose
        self.alpha = (L + 1) // self.dnum  # Number of special primes
        self.K = self.alpha

    def __repr__(self):
        return f"Params: N={self.N}, L={self.L}, dnum={self.dnum}, alpha={self.alpha}, K={self.K}"
    
    def __str__(self):
        return f"Params: N={self.N}, L={self.L}, dnum={self.dnum}, alpha={self.alpha}, K={self.K}"
    
    def get_beta(self, current_level: int) -> int:
        """
        Calculate the beta based on the current levels
        """
        return math.ceil(math.ceil((current_level + 1) / self.alpha))