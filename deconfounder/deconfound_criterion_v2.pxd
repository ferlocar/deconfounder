cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

cdef struct BoundaryRecord:
    double bias
    double r_t
    double r_u
    double weighted_r

cdef class DeconfoundCriterionV2(Criterion):
    cdef double r_u_all_total # reward if we untreat all individuals
    cdef double r_u_all_left
    cdef double r_u_all_right
    cdef double[2] sum_total_arr
    cdef double[2] sum_left_arr
    cdef double[2] sum_right_arr
    cdef double[2] weighted_n_node_arr      # sum the sample weights
    cdef double[2] weighted_n_left_arr
    cdef double[2] weighted_n_right_arr
    cdef double[:] sorted_predictions
    cdef SIZE_t[:] sorted_samples

    cdef int[:] treated  # Defines which observations were treated
    cdef double[:] predictions # Predicted effects by the observational model
    cdef double largest_y # Defines largest y observed (important so that impurity is always positive)
    cdef double p_t    # fraction of treated individuals in the entire data
    cdef double p_u    # fraction of untreated individual in the entire data

    # Replicate variables for the TREATED

    cdef BoundaryRecord decision_boundary(self, double r_t_all, SIZE_t start, SIZE_t end) nogil
    cdef double get_impurity(self, double r_u_all, SIZE_t start, SIZE_t end, double weighted_n_samples) nogil