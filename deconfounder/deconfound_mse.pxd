cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

cdef class DeconfoundMSE(Criterion):
    cdef double[2] sq_sum_node_arr
    cdef double[2] sq_sum_left_arr
    cdef double[2] sq_sum_right_arr
    cdef double[2] sum_node_arr
    cdef double[2] sum_left_arr
    cdef double[2] sum_right_arr
    cdef double[2] weighted_n_node_arr
    cdef double[2] weighted_n_left_arr
    cdef double[2] weighted_n_right_arr
    cdef double sq_sum_node_scores
    cdef double sq_sum_left_scores
    cdef double sq_sum_right_scores
    cdef double sum_node_scores
    cdef double sum_left_scores
    cdef double sum_right_scores

    cdef int[:] treated  # Defines which observations were treated
    cdef double[:] scores

    cdef double get_impurity(self, double[2] sq_sum_arr, double[2] sum_arr, double sq_sum_scores, double sum_scores, double[2] weighted_n_arr) nogil