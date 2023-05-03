cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

cdef struct BoundaryRecord:
    double bias
    double r_t
    double r_u
    double weighted_r

cdef class DeconfoundCriterionV2(Criterion):
    cdef double r_u_all_total   # reward if we untreat all individuals
    cdef double r_u_all_left    # reward if we untreat all individuals in th left node
    cdef double r_u_all_right   # reward if we untreat all individuals in the right node

    cdef int[:] mask_total  # indicate which samples are in the node
    cdef int[:] mask_left   # indicate which samples are in the left node
    cdef int[:] mask_right  # indicate which samples are in the right node

    cdef int[:] treated  # Defines which observations were treated
    cdef double[:] predictions # Predicted effects by the observational model
    cdef double largest_y # Defines largest y observed (important so that impurity is always positive)
    cdef double p_t    # fraction of treated individuals in the entire data
    cdef double p_u    # fraction of untreated individual in the entire data

    # Replicate variables for the TREATED

    cdef BoundaryRecord decision_boundary(self, double r_t_all, int[:] mask) nogil
    cdef double get_impurity(self, double r_u_all, int[:] mask, double weighted_n_samples) nogil