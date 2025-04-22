class Params:
    """
    Class to hold the parameters for the encryption scheme.
    """

    def __init__(self, N: int, L: int):
        self.N = N  # Degree of the polynomial
        self.L = L  # Number of max levels

    def __repr__(self):
        return f"Params(N={self.N}, L={self.L})"
    
    def __str__(self):
        return f"Params(N={self.N}, L={self.L})"