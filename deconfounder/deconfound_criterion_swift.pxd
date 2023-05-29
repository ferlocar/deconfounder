cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DTYPE_t

cdef struct BoundaryRecord:
    double threshold
    double reward

cdef class DeconfoundCriterion(Criterion):
    cdef double r_u_all_node   # reward if we untreat all individuals in the node
    cdef double[2] r_u_all_children    # reward if we untreat all individuals in th children node

    cdef SIZE_t[::1] sorted_samples   # Splitted by node in which samples are sorted by predictions
    cdef int[:] children_mask  # Indicate if the observation is left or right

    cdef int[:] treated  # Defines which observations were treated
    cdef double[:] predictions # Predicted effects by the observational model
    cdef double[:] cost   # Cost of treatment to individuals
    cdef double p_t    # fraction of treated individuals in the entire data
    cdef double p_u    # fraction of untreated individual in the entire data

    cdef BoundaryRecord node_decision_boundary(self) nogil
    cdef BoundaryRecord* children_decision_boundary(self) nogil

