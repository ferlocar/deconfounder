from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t
import numpy as np

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport atan, tan
from libc.stdio cimport printf

cdef int U_IX = 0
cdef int T_IX = 1
cdef double INFINITY = 1e9


cdef class CausalResidualMSE(Criterion):
    """
    This criterion seeks to minimize the MSE of residuals, defined as the difference between base scores and CATE.
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
        
        # Default values
        # self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Additional variables
        self.sq_score_sum_total = 0.0
        self.sq_score_sum_left = 0.0
        self.sq_score_sum_right = 0.0
        self.score_sum_total = 0.0
        self.score_sum_left = 0.0
        self.score_sum_right = 0.0

        cdef int t_i

        for t_i in range(2):
            self.sq_sum_node_arr[t_i] = 0.0
            self.sq_sum_left_arr[t_i] = 0.0
            self.sq_sum_right_arr[t_i] = 0.0
            self.sum_node_arr[t_i] = 0.0
            self.sum_left_arr[t_i] = 0.0
            self.sum_right_arr[t_i] = 0.0
            self.weighted_n_node_arr[t_i] = 0.0
            self.weighted_n_right_arr[t_i] = 0.0
            self.weighted_n_left_arr[t_i] = 0.0

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    def __getstate__(self):
        d = {}
        d['treated'] = np.asarray(self.treated)
        d['scores'] = np.asarray(self.scores)
        return d

    def __setstate__(self, d):
        self.treated = np.asarray(d['treated'])
        self.scores = np.asarray(d['scores'])

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

        self.sq_score_sum_total = 0.
        self.score_sum_total = 0.

        cdef int t_i
        for t_i in range(2):
            self.sq_sum_node_arr[t_i] = 0.0
            self.sum_node_arr[t_i] = 0.0
            self.weighted_n_node_arr[t_i] = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef double score_i


        for p in range(start, end):
            i = samples[p]
            t_i = self.treated[i]
            score_i = self.scores[i]
            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sq_sum_node_arr[t_i] += w_y_ik * y_ik
                self.sum_node_arr[t_i] += w_y_ik
                self.sq_score_sum_total += score_i ** 2
                self.score_sum_total += score_i

            self.weighted_n_node_arr[t_i] += w

        self.weighted_n_node_samples = self.weighted_n_node_arr[U_IX] + self.weighted_n_node_arr[T_IX]

        # Reset to pos=start
        self.reset()

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef int t_i

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        self.sq_score_sum_left = 0.0
        self.sq_score_sum_right = self.sq_score_sum_total

        self.score_sum_left = 0.0
        self.score_sum_right = self.score_sum_total

        for t_i in range(2):
            self.sq_sum_left_arr[t_i] = 0.0
            self.sq_sum_right_arr[t_i] = self.sq_sum_node_arr[t_i]
            self.sum_left_arr[t_i] = 0.0
            self.sum_right_arr[t_i] = self.sum_node_arr[t_i]
            self.weighted_n_left_arr[t_i] = 0.0
            self.weighted_n_right_arr[t_i] = self.weighted_n_node_arr[t_i]

        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef int t_i

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        self.sq_score_sum_left = self.sq_score_sum_total
        self.sq_score_sum_right = 0.0

        self.score_sum_left = self.score_sum_total
        self.score_sum_right = .0

        for t_i in range(2):
            self.sq_sum_left_arr[t_i] = self.sq_sum_node_arr[t_i]
            self.sq_sum_right_arr[t_i] = 0.0
            self.sum_left_arr[t_i] = self.sum_node_arr[t_i]
            self.sum_right_arr[t_i] = 0.0
            self.weighted_n_left_arr[t_i] = self.weighted_n_node_arr[t_i]
            self.weighted_n_right_arr[t_i] = 0.0

        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef const DOUBLE_t[:] sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef int t_i
        cdef double score_i

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
                t_i = self.treated[i]
                score_i = self.scores[i]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    w_y_ik = w * y_ik
                    self.sq_sum_left_arr[t_i] += w_y_ik * y_ik
                    self.sum_left_arr[t_i] += w_y_ik
                    self.sq_score_sum_left += score_i ** 2
                    self.score_sum_left += score_i

                self.weighted_n_left_arr[t_i] += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                t_i = self.treated[i]
                score_i = self.scores[i]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    w_y_ik = w * y_ik
                    self.sq_sum_left_arr[t_i] -= w_y_ik * y_ik
                    self.sum_left_arr[t_i] -= w_y_ik
                    self.sq_score_sum_left -= score_i**2
                    self.score_sum_left -= score_i

                self.weighted_n_left_arr[t_i] -= w

        for t_i in range(2):
            self.sum_right_arr[t_i] = self.sum_node_arr[t_i] - self.sum_left_arr[t_i]
            self.sq_sum_right_arr[t_i] = self.sq_sum_node_arr[t_i] - self.sq_sum_left_arr[t_i]
            self.weighted_n_right_arr[t_i] = self.weighted_n_node_arr[t_i] - self.weighted_n_left_arr[t_i]
        
        self.sq_score_sum_right = self.sq_score_sum_total - self.sq_score_sum_left
        self.score_sum_right = self.score_sum_total - self.score_sum_left
        self.weighted_n_left = self.weighted_n_left_arr[U_IX] + self.weighted_n_left_arr[T_IX]
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left

        self.pos = new_pos
        return 0


    cdef double get_impurity(self, double[2] sq_sum_arr, double[2] sum_arr, double 
                                sq_score_sum, double score_sum, double[2] weighted_n_arr) nogil:
            
        cdef double impurity = 0
        # cdef double variance
        cdef double resid
        cdef double avg_score, avg_t, avg_u
        cdef int t_i
        cdef bint is_missing = False
        cdef double weighted_n_samples = weighted_n_arr[U_IX] + weighted_n_arr[T_IX]

        # variance = sq_score_sum / weighted_n_samples
        # variance -= (score_sum / weighted_n_samples)**2.0
        # variance /= weighted_n_samples
        # impurity += variance

        for t_i in range(2):
            if weighted_n_arr[t_i] > 0:
                pass
                # variance = sq_sum_arr[t_i] / weighted_n_arr[t_i]
                # variance -= (sum_arr[t_i] / weighted_n_arr[t_i])**2.0
                # variance /= weighted_n_arr[t_i]
                # impurity += variance
            else:
                is_missing = True

        if is_missing:
            # In the case of missing data, the impurity should be very large so the split is not done
            impurity = INFINITY
        else:
            # Heterogeneity in treatments
            avg_score = score_sum / weighted_n_samples
            avg_t = sum_arr[T_IX] / weighted_n_arr[T_IX]
            avg_u = sum_arr[U_IX] / weighted_n_arr[U_IX]
            resid = avg_t - avg_u - avg_score
            impurity -= resid**2

        # The SKLEARN tree requires impurity to be non-negative
        impurity = calc_atan(impurity)

        return impurity

    
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double impurity = 0
        cdef double impurity_k

        for k in range(self.n_outputs):
            impurity_k = self.get_impurity(self.sq_sum_node_arr, self.sum_node_arr, self.sq_score_sum_total, 
                                            self.score_sum_total, self.weighted_n_node_arr)
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
            impurity_left[0] += self.get_impurity(self.sq_sum_left_arr, self.sum_left_arr, self.sq_score_sum_left, 
                                                    self.score_sum_left, self.weighted_n_left_arr)
            impurity_right[0] += self.get_impurity(self.sq_sum_right_arr, self.sum_right_arr, self.sq_score_sum_right, 
                                                    self.score_sum_right, self.weighted_n_right_arr)

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:

        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * calc_tan(impurity_right)
                - self.weighted_n_left * calc_tan(impurity_left))

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - calc_atan(
                    (self.weighted_n_right / self.weighted_n_node_samples * calc_tan(impurity_right))
                    + (self.weighted_n_left / self.weighted_n_node_samples * calc_tan(impurity_left)))
                ))

    cdef void node_value(self, double * dest) nogil:
        """
        - Compute the node value of samples[start:end] into dest.
        - The node value is the difference between average score and ATE in the node.
        """

        cdef SIZE_t k
        cdef double avg_score, avg_t, avg_u

        for k in range(self.n_outputs):
            # The max is included as quick-ft_i for zero division
            avg_score = self.score_sum_total / max(1, self.weighted_n_node_samples)
            avg_t = self.sum_node_arr[T_IX] / max(1, self.weighted_n_node_arr[T_IX])
            avg_u = self.sum_node_arr[U_IX] / max(1, self.weighted_n_node_arr[U_IX])
            dest[k] = avg_t - avg_u - avg_score

    def set_sample_parameters(self, int[:] treated, double[:] scores):
        self.treated = treated
        self.scores = scores


cdef inline double calc_atan(double x) nogil:
    return atan(x)+2

cdef inline double calc_tan(double x) nogil:
    return tan(x-2)