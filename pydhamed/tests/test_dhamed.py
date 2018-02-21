import numpy as np
from pydhamed.optimize_dhamed import *
from pydhamed.util.testing import data

from numpy.testing import assert_almost_equal


def test_cg_rna_pmf_with_reference(data):
    cg_rna_ref = np.genfromtxt(data["us_dt-e4_2_p1.out"])
    c_l = [np.genfromtxt(data["count_matrix_1.txt"])]
    v_ar = np.genfromtxt(data["wfile.txt"])[:,1].reshape((9,1))

    og = run_dhamed(c_l, -np.log(v_ar), g_init=np.zeros(9), maxiter=10000)
    py_rna = og*-1 - (og[-1]*-1)
    f_rna = cg_rna_ref[:,-1] - cg_rna_ref[-1,-1]

    assert_almost_equal(f_rna, py_rna)
