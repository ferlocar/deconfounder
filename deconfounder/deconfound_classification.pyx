from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DTYPE_t, DOUBLE_t
import numpy as np

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport atan, tan
from libc.stdio cimport printf
from libc.math cimport log as ln


cdef int U_IX = 0
cdef int T_IX = 1
cdef int L_IX = 1
cdef int R_IX = 0
cdef double SCORE_THRESHOLD = 1e-7
cdef double INFINITY = 1e9


cdef class DeconfoundClassification(Criterion):
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
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sorted_samples = np.empty(n_samples, dtype=np.int64)
        self.children_mask = np.zeros(n_samples, dtype=np.int32)

        for is_treated in range(2):
            self.weighted_n_node_arr[is_treated] = 0.0
            self.weighted_n_left_arr[is_treated] = 0.0
            self.weighted_n_right_arr[is_treated] = 0.0
            self.sum_node_arr[is_treated] = .0
            self.sum_left_arr[is_treated] = .0
            self.sum_right_arr[is_treated] = .0

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())
    
    def __getstate__(self):
        d = {}
        d['treated'] = np.asarray(self.treated)
        d['scores'] = np.asarray(self.scores)
        d['cost'] = np.asarray(self.cost)
        return d

    def __setstate__(self, d):
        self.treated = np.asarray(d['treated'])
        self.scores = np.asarray(d['scores'])
        self.cost = np.asarray(d['cost'])

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

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef int is_treated

        cdef SIZE_t[::1] sorted_samples = self.sorted_samples

        for is_treated in range(2):
            self.weighted_n_node_arr[is_treated] = 0.0
            self.sum_node_arr[is_treated] = 0.0

        for p in range(start, end):
            i = samples[p]
            is_treated = self.treated[i]
            
            if sample_weight is not None:
                w = sample_weight[i]
            
            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik

                self.sum_node_arr[is_treated] += w_y_ik

            self.weighted_n_node_arr[is_treated] += w

            insert_sorted(&sorted_samples[start], i, p-start)

        self.weighted_n_node_samples = self.weighted_n_node_arr[U_IX] + self.weighted_n_node_arr[T_IX]
            
        # Reset to pos=start
        self.reset()

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t p, i
        cdef int is_treated

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        for is_treated in range(2):
            self.weighted_n_left_arr[is_treated] = 0.0
            self.weighted_n_right_arr[is_treated] = self.weighted_n_node_arr[is_treated]
            self.sum_left_arr[is_treated] = .0
            self.sum_right_arr[is_treated] = self.sum_node_arr[is_treated]
        
        for p in range(self.start, self.end):
            i = self.samples[p]
            self.children_mask[i] = R_IX

        self.pos = self.start

        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t p, i
        cdef int is_treated

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        for is_treated in range(2):
            self.weighted_n_left_arr[is_treated] = self.weighted_n_node_arr[is_treated]
            self.weighted_n_right_arr[is_treated] = 0.0
            self.sum_left_arr[is_treated] = self.sum_node_arr[is_treated]
            self.sum_right_arr[is_treated] = .0

        for p in range(self.start, self.end):
            i = self.samples[p]
            self.children_mask[i] = L_IX

        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef const DOUBLE_t[:] sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef double[:] scores = self.scores

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

                    self.sum_left_arr[is_treated] += w_y_ik

                self.weighted_n_left_arr[is_treated] += w
                self.children_mask[i] = L_IX

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

                    self.sum_left_arr[is_treated] -= w_y_ik

                self.weighted_n_left_arr[is_treated] -= w
                self.children_mask[i] = R_IX
    
        for is_treated in range(2):
            self.weighted_n_right_arr[is_treated] = self.weighted_n_node_arr[is_treated] - self.weighted_n_left_arr[is_treated]
            self.sum_right_arr[is_treated] = self.sum_node_arr[is_treated] - self.sum_left_arr[is_treated]

        self.weighted_n_left = self.weighted_n_left_arr[U_IX] + self.weighted_n_left_arr[T_IX]
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left

        self.pos = new_pos

        return 0

    cdef BoundaryRecord init_boundary(self, double max_score, double[2] sum_arr, double[2] weighted_n_arr) nogil:

        cdef BoundaryRecord br
        br.threshold = max_score + SCORE_THRESHOLD
        br.p_t = weighted_n_arr[T_IX] / (weighted_n_arr[U_IX] + weighted_n_arr[T_IX])
        br.p_u = 1 - br.p_t
        br.reward = sum_arr[U_IX] / max(1, br.p_u)
        br.tmf = weighted_n_arr[U_IX] /max(1, br.p_u)
        return br


    cdef BoundaryRecord node_decision_boundary(self) nogil:
        
        cdef BoundaryRecord curr, best

        cdef double[:] scores = self.scores
        cdef double w = 1.0
        cdef double y_i
        cdef double cost_i

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t p, i

        i = self.sorted_samples[start]
        curr = self.init_boundary(scores[i], self.sum_node_arr, self.weighted_n_node_arr)
        best = curr

        for p in range(start, end):
            i = self.sorted_samples[p]

            # Calculate total reward when we treat everyone with a prediction greater than scores[i].
            if (curr.threshold > scores[i]):      
                if (curr.reward * best.tmf) > (best.reward * curr.tmf):
                    best = curr
                # Update bias so that observation i is treated
                curr.threshold = scores[i] - SCORE_THRESHOLD

            # Update reward when observation i is treated
            if self.sample_weight is not None:
                w = self.sample_weight[i]
            y_i = self.y[i, 0] 
            cost_i = self.cost[i]

            if self.treated[i]:
                curr.reward += w * (y_i - cost_i) / curr.p_t   # benefit - cost
                curr.tmf += w / curr.p_t
            else:
                curr.reward -= w * y_i / curr.p_u
                curr.tmf -= w / curr.p_u

        if (curr.reward * best.tmf) > (best.reward * curr.tmf):
            best = curr

        return best

    cdef BoundaryRecord* children_decision_boundary(self) nogil:
        
        cdef BoundaryRecord[2] curr, best

        cdef double[:] scores = self.scores
        cdef double w = 1.0
        cdef double y_i
        cdef double cost_i

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef int is_left
        cdef SIZE_t p, i

        i = self.sorted_samples[start]
        curr[L_IX] = self.init_boundary(scores[i], self.sum_left_arr, self.weighted_n_left_arr)
        best[L_IX] = curr[L_IX]
        curr[R_IX] = self.init_boundary(scores[i], self.sum_right_arr, self.weighted_n_right_arr)
        best[R_IX] = curr[R_IX]

        for p in range(start, end):
            i = self.sorted_samples[p]
            is_left = self.children_mask[i]

            # Calculate total reward when we treat everyone with a prediction greater than scores[i].
            if (curr[is_left].threshold > scores[i]):      
                if (curr[is_left].reward * best[is_left].tmf) > (best[is_left].reward * curr[is_left].tmf):
                    best[is_left] = curr[is_left]
                # Update bias so that observation i is treated
                curr[is_left].threshold = scores[i] - SCORE_THRESHOLD

            # Update reward when observation i is treated
            if self.sample_weight is not None:
                w = self.sample_weight[i]
            y_i = self.y[i, 0]
            cost_i = self.cost[i]

            if self.treated[i]:
                curr[is_left].reward += w * (y_i - cost_i) / curr[is_left].p_t     # benefit - cost
                curr[is_left].tmf += w / curr[is_left].p_t
            else:
                curr[is_left].reward -= w * y_i / (curr[is_left].p_u)
                curr[is_left].tmf -= w / (curr[is_left].p_u)

        for is_left in range(2):
            if (curr[is_left].reward * best[is_left].tmf) > (best[is_left].reward * curr[is_left].tmf):
                best[is_left] = curr[is_left]

        return best

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double impurity = 0

        for k in range(self.n_outputs):
            boundary = self.node_decision_boundary()
            impurity -= boundary.reward / max(1, boundary.tmf)

        impurity /= self.n_outputs

        return calc_atan(impurity)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        impurity_left[0] = 0
        impurity_right[0] = 0

        if self.weighted_n_left_arr[U_IX]==0 or self.weighted_n_left_arr[T_IX]==0 \
            or self.weighted_n_right_arr[U_IX]==0 or self.weighted_n_right_arr[T_IX]==0:
            impurity_left[0] = INFINITY
            impurity_right[0] = INFINITY
        else:
            for k in range(self.n_outputs):
                children_boundary = self.children_decision_boundary()
                impurity_left[0] -= children_boundary[L_IX].reward / max(1, children_boundary[L_IX].tmf)
                impurity_right[0] -= children_boundary[R_IX].reward / max(1, children_boundary[R_IX].tmf)

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
        impurity_left[0] = calc_atan(impurity_left[0])
        impurity_right[0] = calc_atan(impurity_right[0])


    cdef void node_value(self, double * dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k
        cdef BoundaryRecord boundary

        for k in range(self.n_outputs):
            boundary = self.node_decision_boundary()
            dest[k] = boundary.threshold

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

    def set_sample_parameters(self, int[:] treated, double[:] scores, double[:] cost):
        self.treated = treated
        self.scores = scores
        self.cost = cost

cdef inline double calc_atan(double x) nogil:
    return atan(x)+2

cdef inline double calc_tan(double x) nogil:
    return tan(x-2)

# Binary search for the position of x in an array of length n
cdef inline SIZE_t bisect_left(SIZE_t* a, SIZE_t x, SIZE_t lo, SIZE_t hi) nogil:

    cdef SIZE_t mid

    if (lo < 0):
        printf("lo must be non-negative\n")
        return -1

    while (lo < hi):
        mid = (lo + hi) / 2
        if (a[mid] < x):
            lo = mid + 1
        else:
            hi = mid
    return lo

# Insert x into an array of length n
cdef inline void insert_sorted(SIZE_t* a, SIZE_t x, SIZE_t n) nogil:

    cdef SIZE_t j, k
    k = bisect_left(a, x, 0, n)

    j = n-1
    while j >= k:
        a[j+1] = a[j]
        j -= 1
    a[k] = x



