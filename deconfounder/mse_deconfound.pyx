from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t
import numpy as np

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.stdio cimport printf

cdef int U_OBS_IX = 0
cdef int T_OBS_IX = 1
cdef int U_EXP_IX = 2
cdef int T_EXP_IX = 3


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
        cdef int ix

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

        for ix in range(4):
            self.sq_sum_total_arr[ix] = 0.0
            self.sq_sum_left_arr[ix] = 0.0
            self.sq_sum_right_arr[ix] = 0.0
            self.sum_total_arr[ix] = 0.0
            self.sum_left_arr[ix] = 0.0
            self.sum_right_arr[ix] = 0.0
            self.weighted_n_node_arr[ix] = 0.0
            self.weighted_n_right_arr[ix] = 0.0
            self.weighted_n_left_arr[ix] = 0.0

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    def __getstate__(self):
        d = {}
        d['treated'] = np.asarray(self.treated)
        d['experiment'] = np.asarray(self.experiment)
        return d

    def __setstate__(self, d):
        self.treated = np.asarray(d['treated'])
        self.experiment = np.asarray(d['experiment'])

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

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef int is_treated
        cdef int is_experiment
        cdef int ix

        for ix in range(4):
            self.sq_sum_total_arr[ix] = 0.0
            self.sum_total_arr[ix] = 0.0
            self.weighted_n_node_arr[ix] = 0.0

        for p in range(start, end):
            i = samples[p]
            is_treated = self.treated[i]
            is_experiment = self.experiment[i]
            ix = is_treated + 2*is_experiment
            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sq_sum_total_arr[ix] += w_y_ik * y_ik
                self.sum_total_arr[ix] += w_y_ik

                if self.largest_y < fabs(y_ik):
                    self.largest_y = fabs(y_ik)

            self.weighted_n_node_arr[ix] += w

        for ix in range(4):
            self.weighted_n_node_samples += self.weighted_n_node_arr[ix]

        # Reset to pos=start
        self.reset()

        for ix in range(4):
            self.weighted_n_left = self.weighted_n_left_arr[ix]
            self.weighted_n_right = self.weighted_n_right_arr[ix]
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef int ix

        for ix in range(4):
            self.sum_left_arr[ix] = 0.0
            self.sq_sum_left_arr[ix] = 0.0
            self.sum_right_arr[ix] = self.sum_total_arr[ix]
            self.sq_sum_right_arr[ix] = self.sq_sum_total_arr[ix]
            self.weighted_n_left_arr[ix] = 0.0
            self.weighted_n_right_arr[ix] = self.weighted_n_node_arr[ix]

        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef int ix

        for ix in range(4):
            self.sum_left_arr[ix] = self.sum_total_arr[ix]
            self.sq_sum_left_arr[ix] = self.sq_sum_total_arr[ix]
            self.sum_right_arr[ix] = 0.0
            self.sq_sum_right_arr[ix] = 0.0
            self.weighted_n_left_arr[ix] = self.weighted_n_node_arr[ix]
            self.weighted_n_right_arr[ix] = 0.0

        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left_arr
        cdef double* sum_right = self.sum_right_arr
        cdef double* sum_total = self.sum_total_arr

        cdef double* sq_sum_left = self.sq_sum_left_arr
        cdef double* sq_sum_right = self.sq_sum_right_arr
        cdef double* sq_sum_total = self.sq_sum_total_arr

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
        cdef int is_treated
        cdef int is_experiment
        cdef int ix

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
                is_experiment = self.experiment[i]
                ix = is_treated + 2 * is_experiment

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    w_y_ik = w * y_ik
                    sum_left[ix] += w_y_ik
                    sq_sum_left[ix] += w_y_ik * y_ik

                self.weighted_n_left_arr[ix] += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                is_treated = self.treated[i]
                is_experiment = self.experiment[i]
                ix = is_treated + 2 * is_experiment

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    w_y_ik = w * y_ik
                    sum_left[ix] -= w_y_ik
                    sq_sum_left[ix] -= w_y_ik * y_ik

                self.weighted_n_left_arr[ix] -= w
        self.weighted_n_left = 0
        for ix in range(4):
            sum_right[ix] = sum_total[ix] - sum_left[ix]
            sq_sum_right[ix] = sq_sum_total[ix] - sq_sum_left[ix]
            self.weighted_n_right_arr[ix] = self.weighted_n_node_arr[ix] - self.weighted_n_left_arr[ix]
            self.weighted_n_left += self.weighted_n_left_arr[ix]
        self.weighted_n_right = (self.weighted_n_node_samples - self.weighted_n_left)

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double impurity = 0
        cdef double impurity_k

        for k in range(self.n_outputs):
            impurity_k = self.get_impurity(self.sq_sum_total_arr, self.sum_total_arr, self.weighted_n_node_arr)
            impurity += impurity_k

        return impurity / self.n_outputs

    cdef double get_impurity(self, double* sq_sum_total, double* sum_total, double* sample_weight) nogil:
        cdef double impurity = 0
        cdef double variance
        cdef double bias
        cdef double obs_effect
        cdef double exp_effect
        cdef int ix
        cdef int missing_data = 0

        for ix in range(4):
            # Variance in the estimates (s^2/N_u)
            if sample_weight[ix] > 0:
                variance = sq_sum_total[ix] / sample_weight[ix]
                variance -= (sum_total[ix] / sample_weight[ix])**2.0
                variance /= sample_weight[ix]
                impurity += variance
            else:
                missing_data = 1

        if missing_data:
            # In the case of missing data, the impurity should be very large so the split is not done
            impurity += self.largest_y ** 2 + 1
        else:
            # Heterogeneity in treatments
            obs_effect = sum_total[T_OBS_IX] / sample_weight[T_OBS_IX]
            obs_effect -= sum_total[U_OBS_IX] / sample_weight[U_OBS_IX]
            exp_effect = sum_total[T_EXP_IX] / sample_weight[T_EXP_IX]
            exp_effect -= sum_total[U_EXP_IX] / sample_weight[U_EXP_IX]
            bias = obs_effect - exp_effect
            impurity -= bias**2

        # The SKLEARN tree requires impurity to be greater than 0.
        # So, we sum this constant to make sure it does.
        impurity += self.largest_y ** 2 + 1

        # Constant so that impurity is positive
        return impurity


    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        impurity_left[0] = 0
        impurity_right[0] = 0

        for k in range(self.n_outputs):
            impurity_left[0] += self.get_impurity(self.sq_sum_left_arr, self.sum_left_arr, self.weighted_n_left_arr)
            impurity_right[0] += self.get_impurity(self.sq_sum_right_arr, self.sum_right_arr, self.weighted_n_right_arr)

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs

    cdef void node_value(self, double * dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k
        cdef double avg_t
        cdef double avg_u

        for k in range(self.n_outputs):
            # The max is included as quick-fix for zero division
            avg_t_obs = self.sum_total_arr[T_OBS_IX] / max(1, self.weighted_n_node_arr[T_OBS_IX])
            avg_u_obs = self.sum_total_arr[U_OBS_IX] / max(1, self.weighted_n_node_arr[U_OBS_IX])
            avg_t_exp = self.sum_total_arr[T_EXP_IX] / max(1, self.weighted_n_node_arr[T_EXP_IX])
            avg_u_exp = self.sum_total_arr[U_EXP_IX] / max(1, self.weighted_n_node_arr[U_EXP_IX])
            dest[k] = (avg_t_obs - avg_u_obs) - (avg_t_exp - avg_u_exp)

    def set_treated_experiment(self, int[:] treated, int[:] experiment):
        self.treated = treated
        self.experiment = experiment