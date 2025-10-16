cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

cdef class CausalResidualMSE(Criterion):
    cdef double[2] sq_sum_node_arr          # Sum of squared Yt or Yc.
    cdef double[2] sq_sum_left_arr           
    cdef double[2] sq_sum_right_arr                  
    cdef double[2] sum_node_arr             # Sum of Yt or Yc.
    cdef double[2] sum_left_arr
    cdef double[2] sum_right_arr
    cdef double[2] weighted_n_node_arr      # Sum of weights of treated or control units.
    cdef double[2] weighted_n_left_arr
    cdef double[2] weighted_n_right_arr
    cdef double sq_score_sum_total          # Sum of squared scores.
    cdef double sq_score_sum_left
    cdef double sq_score_sum_right
    cdef double score_sum_total             # Sum of scores
    cdef double score_sum_left
    cdef double score_sum_right

    cdef int[:] treated  # Defines which observations were treated
    cdef double[:] scores

    cdef double get_impurity(self, double[2] sq_sum_arr, double[2] sum_arr, double sq_sum_scores, double sum_scores, double[2] weighted_n_arr) nogil