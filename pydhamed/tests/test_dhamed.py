import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from optimize_dhamed import *

class TestDHAMed:
      def __init__(self):
          #cg_rna_ref = np.genfromtxt("../cg-rna/")
          pass
      

      def test_cg_rna_pmf_with_reference(self):
          cg_rna_ref = np.genfromtxt("test_rna/us_dt-e4_2_p1.out")
          c_l = [np.genfromtxt("test_rna/count_matrix_1.txt")]
          v_ar = np.genfromtxt("test_rna/wfile.txt")[:,1].reshape((9,1))
          og = run_dhamed(c_l, -np.log(v_ar), g_init=-(np.zeros(9))*-1.0, maxiter=10000)
          py_rna = og*-1 - (og[-1]*-1)
          f_rna = cg_rna_ref[:,-1] - cg_rna_ref[-1,-1]
          np.testing.assert_almost_equal(f_rna, py_rna)
          fig, ax = plt.subplots(figsize=(5,3))
          plt.plot(cg_rna_ref[:,0], py_rna, "o", label="pyDHAMed")
          plt.plot(cg_rna_ref[:,0],f_rna, label="ref calculation")
          ax.set_xlabel("Reaction coordinate")
          ax.set_ylabel("PMF ($\mathrm{k_BT}$)")
          ax.legend()
          fig.tight_layout() 
          fig.savefig("test_rna/rna_pmf.png")
      
