"""
A script for simplifying equations using the sympy package
"""

from sympy import *


if __name__ == "__main__":

    # deriving the equation used in parity_estimation_independent_proxy
    # delta1, s, p, q = symbols('delta1 s p q')
    # denom = (s - s * q + p - p * s) * (q * s + 1 - s - p + p * s)
    # A1 = (s * (1-s)) / denom
    # A2 = (2 * s * (s - 1) + 1) / denom
    # B1 = (1 - p) * (1 - q)
    # B2 = (p * (1 - p) * (1 - s) ** 2 + q * (1-q) * s ** 2) / (2 * s * (s - 1) + 1)
    # B3 = p * q
    # exp = (delta1 - A2 * B2 - 2 * A1 * B3 + 1) / (A1 * B1 - A1 * B3) - 1
    # print(simplify(expand(exp)))
    # print(latex(simplify(expand(exp))))
    # print(latex(simplify(exp.subs([(p, q), (s, 0.5)]))))
    # # print(simplify(exp.subs([(p, 1-p), (q, 1 - q)])))

    # test parity_estimation_independent_proxy
    # res = parity_estimation_independent_proxy(0.5, 0.25, 0.25, 0.5)
    # print(res)

    m, p, q = symbols('m p q')
    A = p ** 2 * m ** 2 + \
        2 * p **2 * m - \
        p ** 2 - \
        2 * p * q * m ** 2 + \
        2 * p * q * m - \
        3 * p * m + \
        p - \
        q ** 2 * m ** 2 + \
        2 * q * m ** 2 - \
        q * m - m ** 2 + m
    A = 2 * p * q * m - 2 * p * m - q * m + m + p - 2 * p ** 2 - p ** 2 * m - p ** 2
    print(factor(A))
