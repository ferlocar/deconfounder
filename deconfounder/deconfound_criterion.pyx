from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t
import numpy as np

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.stdio cimport printf
from libc.math cimport log as ln


cdef int U_IX = 0
cdef int T_IX = 1
cdef double INFINITY = np.inf
cdef double PRED_THRESHOLD = 1e-7


cdef class DeconfoundCriterion(Criterion):
    r"""Abstract regression criterion.
    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::
        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """
        cdef int is_treated

        # Default values
        # self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        # IMPORTANT: The causal tree always assumes that there is only one output.
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.r_u_all_total = 0.0
        self.r_u_all_left = 0.0
        self.r_u_all_right = 0.0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.mask_total = np.zeros(n_samples, dtype=np.int32)
        self.mask_left = np.zeros(n_samples, dtype=np.int32)
        self.mask_right = np.zeros(n_samples, dtype=np.int32)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())
    
    def __getstate__(self):
        d = {}
        d['treated'] = np.asarray(self.treated)
        d['predictions'] = np.asarray(self.predictions)
        d['p_t'] = self.p_t
        d['p_u'] = self.p_u
        return d

    def __setstate__(self, d):
        self.treated = np.asarray(d['treated'])
        self.predictions = np.asarray(d['predictions'])
        self.p_t = d['p_t']
        self.p_u = d['p_u']

    cdef int init(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:] sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        self.largest_y = fabs(y[0, 0])
        self.r_u_all_total = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef int is_treated

        for i in range(self.n_samples):
            self.mask_total[i] = 0

        for p in range(start, end):
            i = samples[p]
            is_treated = self.treated[i]
            
            if sample_weight is not None:
                w = sample_weight[i]
            
            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik

                if not is_treated:
                    self.r_u_all_total += w_y_ik

                if self.largest_y < fabs(y_ik):
                    self.largest_y = fabs(y_ik)

            self.weighted_n_node_samples += w
            self.mask_total[i] = 1
        
        # Reset to pos=start
        self.reset()

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t i

        self.r_u_all_left = 0.0
        self.r_u_all_right = self.r_u_all_total
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        
        for i in range(self.n_samples):
            self.mask_left[i] = 0
            self.mask_right[i] = self.mask_total[i]
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t i

        self.r_u_all_left = self.r_u_all_total
        self.r_u_all_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        self.pos = self.end

        for i in range(self.n_samples):
            self.mask_left[i] = self.mask_total[i]
            self.mask_right[i] = 0
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef const DOUBLE_t[:] sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef double[:] predictions = self.predictions

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef int is_treated

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]
                is_treated = self.treated[i]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    w_y_ik = w * y_ik

                    if not is_treated:
                        self.r_u_all_left += w_y_ik

                self.weighted_n_left += w
                self.mask_left[i] = 1
                self.mask_right[i] = 0

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                is_treated = self.treated[i]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    w_y_ik = w * y_ik

                    if not is_treated:
                        self.r_u_all_left -= w_y_ik

                self.weighted_n_left -= w
                self.mask_left[i] = 0
                self.mask_right[i] = 1

        self.r_u_all_right = self.r_u_all_total - self.r_u_all_left
        self.weighted_n_right = (self.weighted_n_node_samples - self.weighted_n_left)

        self.pos = new_pos

        return 0

    cdef BoundaryRecord decision_boundary(self, double r_u_all, int[:] mask) nogil:
        
        cdef BoundaryRecord curr, best
        cdef double w = 1.0
        cdef double w_y_i0
        cdef SIZE_t i

        curr.bias = INFINITY
        curr.r_u = r_u_all
        curr.r_t = 0.0
        curr.weighted_r = r_u_all / self.p_u
        best = curr

        for i in range(self.n_samples):
            if not mask[i]:     # the sample not in the node
                continue

            # check if the current sample has lower prediction value than the previous one, 
            # if do, calculate the reward for this bias (not include the current sample)
            if (curr.bias > self.predictions[i] + PRED_THRESHOLD):      
                curr.weighted_r = curr.r_t / self.p_t + curr.r_u / self.p_u
                if curr.weighted_r > best.weighted_r:
                    best = curr

            if self.sample_weight is not None:
                w = self.sample_weight[i]
            w_y_i0 = w * self.y[i, 0]

            if self.treated[i]:
                curr.r_t += w_y_i0
            else:
                curr.r_u -= w_y_i0

            curr.bias = self.predictions[i]

        curr.weighted_r = curr.r_t / self.p_t + curr.r_u / self.p_u
        if curr.weighted_r > best.weighted_r:
            best = curr

        return best

    cdef double get_impurity(self, double r_u_all, int[:] mask, double weighted_n_samples) nogil:

        cdef double impurity = 0.
        cdef BoundaryRecord boundary

        # decision boundary
        boundary = self.decision_boundary(r_u_all, mask)

        # impurity
        impurity = - boundary.weighted_r / max(1, weighted_n_samples)

        # The SKLEARN tree requires impurity to be greater than 0.
        # So, we sum this constant to make sure it does.
        impurity += self.largest_y / min(self.p_u, self.p_t)

        return impurity

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double impurity = 0
        cdef double impurity_k

        for k in range(self.n_outputs):
            impurity_k = self.get_impurity(self.r_u_all_total, self.mask_total, self.weighted_n_node_samples)
            impurity += impurity_k

        return impurity / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        impurity_left[0] = 0
        impurity_right[0] = 0

        for k in range(self.n_outputs):
            impurity_left[0] += self.get_impurity(self.r_u_all_left, self.mask_left, self.weighted_n_left)
            impurity_right[0] += self.get_impurity(self.r_u_all_right, self.mask_right, self.weighted_n_right)

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


    cdef void node_value(self, double * dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k
        cdef BoundaryRecord boundary

        for k in range(self.n_outputs):
            boundary = self.decision_boundary(self.r_u_all_total, self.mask_total)
            dest[k] = boundary.bias

    def set_additional_parameters(self, int[:] treated, double[:] predictions, double p_t):
        self.treated = treated
        self.predictions = predictions
        self.p_t = p_t      
        self.p_u = 1 - p_t