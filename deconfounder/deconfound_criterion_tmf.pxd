cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DTYPE_t

cdef struct BoundaryRecord:
    double threshold
    double p_t
    double reward
    double tmf

cdef class DeconfoundCriterion(Criterion):

    cdef double[2] weighted_n_node_arr
    cdef double[2] weighted_n_left_arr
    cdef double[2] weighted_n_right_arr
    cdef double[2] sum_node_arr
    cdef double[2] sum_left_arr
    cdef double[2] sum_right_arr

    cdef SIZE_t[::1] sorted_samples   # Splitted by node in which samples are sorted by predictions
    cdef int[:] children_mask  # Indicate if the observation is left or right

    cdef int[:] treated  # Defines which observations were treated
    cdef double[:] scores # Predicted effects by the observational model
    cdef double[:] cost   # Cost of treatment to individuals

    cdef BoundaryRecord init_boundary(self, double max_score, double[2] sum_arr, double[2] weighted_n_arr) nogil
    cdef BoundaryRecord node_decision_boundary(self) nogil
    cdef BoundaryRecord* children_decision_boundary(self) nogil

