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
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.r_u_all_node = 0.0
        self.r_u_all_children[L_IX] = 0.0
        self.r_u_all_children[R_IX] = 0.0

        self.sorted_samples = np.empty(n_samples, dtype=np.int64)
        self.children_mask = np.zeros(n_samples, dtype=np.int32)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())
    
    def __getstate__(self):
        d = {}
        d['treated'] = np.asarray(self.treated)
        d['predictions'] = np.asarray(self.predictions)
        d['cost'] = np.asarray(self.cost)
        d['p_t'] = self.p_t
        d['p_u'] = self.p_u
        return d

    def __setstate__(self, d):
        self.treated = np.asarray(d['treated'])
        self.predictions = np.asarray(d['predictions'])
        self.cost = np.asarray(d['cost'])
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
        self.r_u_all_node = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0
        cdef int is_treated

        cdef SIZE_t[::1] sorted_samples = self.sorted_samples

        for p in range(start, end):
            i = samples[p]
            is_treated = self.treated[i]
            
            if sample_weight is not None:
                w = sample_weight[i]
            
            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik

                if not is_treated:
                    self.r_u_all_node += w_y_ik

            self.weighted_n_node_samples += w

            insert_sorted(&sorted_samples[start], i, p-start)
            
        # Reset to pos=start
        self.reset()

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t p, i

        self.r_u_all_children[L_IX] = 0.0
        self.r_u_all_children[R_IX] = self.r_u_all_node
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        
        for p in range(self.start, self.end):
            i = self.samples[p]
            self.children_mask[i] = R_IX

        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t i

        self.r_u_all_children[L_IX] = self.r_u_all_node
        self.r_u_all_children[R_IX] = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        self.pos = self.end

        for p in range(self.start, self.end):
            i = self.samples[p]
            self.children_mask[i] = L_IX

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
                    self.r_u_all_children[L_IX] += w_y_ik

                self.weighted_n_left += w
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

                if not is_treated:
                    self.r_u_all_children[L_IX] -= w_y_ik

                self.weighted_n_left -= w
                self.children_mask[i] = R_IX

        self.r_u_all_children[R_IX] = self.r_u_all_node - self.r_u_all_children[L_IX]
        self.weighted_n_right = (self.weighted_n_node_samples - self.weighted_n_left)

        self.pos = new_pos

        return 0

    cdef BoundaryRecord node_decision_boundary(self) nogil:
        
        cdef BoundaryRecord curr, best

        cdef double[:] predictions = self.predictions
        cdef double w = 1.0
        cdef double y_i
        cdef double cost_i

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t p, i

        i = self.sorted_samples[start]
        curr.threshold = predictions[i] + PRED_THRESHOLD
        curr.reward = self.r_u_all_node / self.p_u
        best = curr

        for p in range(start, end):
            i = self.sorted_samples[p]

            # Calculate total reward when we treat everyone with a prediction greater than predictions[i].
            if (curr.threshold > predictions[i]):      
                if curr.reward > best.reward:
                    best = curr
                # Update bias so that observation i is treated
                curr.threshold = predictions[i] - PRED_THRESHOLD

            # Update reward when observation i is treated
            if self.sample_weight is not None:
                w = self.sample_weight[i]
            y_i = self.y[i, 0] 
            cost_i = self.cost[i]

            if self.treated[i]:
                curr.reward += w * (y_i - cost_i) / self.p_t   # benefit - cost
            else:
                curr.reward -= w * y_i / self.p_u

        if curr.reward > best.reward:
            best = curr

        return best

    cdef BoundaryRecord* children_decision_boundary(self) nogil:
        
        cdef BoundaryRecord[2] curr, best

        cdef double[:] predictions = self.predictions
        cdef double w = 1.0
        cdef double y_i
        cdef double cost_i

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef int is_left
        cdef SIZE_t p, i

        i = self.sorted_samples[start]
        for is_left in range(2):
            curr[is_left].threshold = predictions[i] + PRED_THRESHOLD   # largest prediction in the parent node
            curr[is_left].reward = self.r_u_all_children[is_left] / self.p_u
            best[is_left] = curr[is_left]

        for p in range(start, end):
            i = self.sorted_samples[p]
            is_left = self.children_mask[i]

            # Calculate total reward when we treat everyone with a prediction greater than predictions[i].
            if (curr[is_left].threshold > predictions[i]):      
                if curr[is_left].reward > best[is_left].reward:
                    best[is_left] = curr[is_left]
                # Update bias so that observation i is treated
                curr[is_left].threshold = predictions[i] - PRED_THRESHOLD

            # Update reward when observation i is treated
            if self.sample_weight is not None:
                w = self.sample_weight[i]
            y_i = self.y[i, 0]
            cost_i = self.cost[i]

            if self.treated[i]:
                curr[is_left].reward += w * (y_i - cost_i) / self.p_t     # benefit - cost
            else:
                curr[is_left].reward -= w * y_i / self.p_u

        for is_left in range(2):
            if curr[is_left].reward > best[is_left].reward:
                best[is_left] = curr[is_left]

        return best

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double impurity = 0

        for k in range(self.n_outputs):
            boundary = self.node_decision_boundary()
            impurity -= boundary.reward / max(1, self.weighted_n_node_samples)

        impurity = compute_atan(impurity)

        return impurity / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        impurity_left[0] = 0
        impurity_right[0] = 0

        for k in range(self.n_outputs):
            children_boundary = self.children_decision_boundary()
            impurity_left[0] -= children_boundary[L_IX].reward / max(1, self.weighted_n_left)
            impurity_right[0] -= children_boundary[R_IX].reward / max(1, self.weighted_n_right)

        impurity_left[0] = compute_atan(impurity_left[0])
        impurity_right[0] = compute_atan(impurity_right[0])
        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


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

        return (- self.weighted_n_right * compute_tan(impurity_right)
                - self.weighted_n_left * compute_tan(impurity_left))

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (compute_tan(impurity_parent) - (self.weighted_n_right /
                                    self.weighted_n_node_samples * compute_tan(impurity_right))
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * compute_tan(impurity_left))))

    def set_sample_parameters(self, int[:] treated, double[:] predictions, double[:] cost, double p_t):
        self.treated = treated
        self.predictions = predictions
        self.cost = cost
        self.p_t = p_t      
        self.p_u = 1 - p_t

cdef inline double compute_atan(double x) nogil:
    return atan(x)+2

cdef inline double compute_tan(double x) nogil:
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



