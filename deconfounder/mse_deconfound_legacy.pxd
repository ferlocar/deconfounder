cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

cdef class DeconfoundCriterion(Criterion):
    cdef double[4] sq_sum_total_arr
    cdef double[4] sq_sum_left_arr
    cdef double[4] sq_sum_right_arr
    cdef double[4] sum_total_arr
    cdef double[4] sum_left_arr
    cdef double[4] sum_right_arr
    cdef double[4] weighted_n_node_arr
    cdef double[4] weighted_n_left_arr
    cdef double[4] weighted_n_right_arr


    cdef int[:] treated  # Defines which observations were treated
    cdef int[:] experiment  # Defines which observations were randomly treated
    cdef double largest_y # Defines largest y observed (important so that impurity is always positive)

    # Replicate variables for the TREATED


    cdef double get_impurity(self, double* sq_sum_total, double* sum_total, double* sample_weight) nogil