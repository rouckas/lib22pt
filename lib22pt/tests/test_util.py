from lib22pt.util import concentration, langevin
from numpy import isclose

def test_concentration():
    assert isclose(concentration(10, 1e-5, 140.), 1.85133186663e+14)

def test_langevin():
    # normal H2 at 300 K (Milenko et al. 1972):
    # alpha = 0.7923 Angstrom^3
    assert isclose(langevin(0.7923, 15, 2, aunit="A3"), 1.569260623805102e-09)
    # ground state H2 (Kolos and Wolniewicz 1967):
    # alpha = 0.7923 Angstrom^3
    assert isclose(langevin(5.17862, 15, 2, aunit="au"), 1.5443970763383667e-09)
