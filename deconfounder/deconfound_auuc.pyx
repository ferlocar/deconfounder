from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DTYPE_t, DOUBLE_t, int
import numpy as np

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.math cimport log as ln
from libcpp cimport bool

from libcpp.algorithm cimport push_heap
from libcpp.algorithm cimport pop_heap

cdef bint U_IX = 0
cdef bint T_IX = 1

cdef double SCORE_THRESHOLD = 1e-15

BUCKET_DTYPE = np.dtype([
    ('n', np.int32), 
    ('nt',np.int32), 
    ('nu', np.int32), 
    ('sum_yt', np.float64),
    ('sum_yu', np.float64),
    ('score', np.float64)
])

LEVEL_DTYPE = np.dtype([
    ('nt', np.int32), 
    ('nu', np.int32), 
    ('frac', np.float64),
    ('sum_yt', np.float64),
    ('sum_yu', np.float64)
])

cdef inline bool _compare_records(
    const ShiftRecord& left,
    const ShiftRecord& right,
):
    return left.shift > right.shift

cdef inline void _add_to_shift_queue(
    ShiftRecord rec,
    vector[ShiftRecord]& shift_queue,
) nogil:
    """Adds record `rec` to the priority queue `shift_queue`."""

    shift_queue.push_back(rec)
    push_heap(shift_queue.begin(), shift_queue.end(), &_compare_records)


cdef class DeconfoundAUUC(Criterion):

    def __cinit__(self, SIZE_t n_samples, int n_levels, int n_buckets):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """
        # Default values
        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = 1
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_levels = n_levels
        self.n_buckets = n_buckets
        self.n_bkt_per_level = n_buckets / n_levels
        
        # Bucket and level
        self.buckets = np.empty(n_buckets, dtype=BUCKET_DTYPE)
        self.buckets_left = np.empty(n_buckets, dtype=BUCKET_DTYPE)
        self.buckets_right = np.empty(n_buckets, dtype=BUCKET_DTYPE)
        self.levels = np.empty(n_levels, dtype=LEVEL_DTYPE)
        self.curr_levels = np.empty(n_levels, dtype=LEVEL_DTYPE)

        # Left and right
        self.n_bkt_left_level = np.empty(n_levels, dtype=np.int32)
        self.n_bkt_right_level = np.empty(n_levels, dtype=np.int32)

        # Fix and shift
        self.n_bkt_fxd_level = np.empty(n_levels, dtype=np.int32)
        self.n_bkt_sft_level = np.empty(n_levels, dtype=np.int32)
        self.last_fxd_level = np.empty(n_levels, dtype=np.int32)
        self.first_sft_level = np.empty(n_levels, dtype=np.int32)

        self.stump = StumpStruct(pos=0, shift_left=0, shift_right=0, auuc_incr=0)

    def __reduce__(self):
        return (type(self), (self.n_samples, self.n_levels, self.n_buckets), \
                self.__getstate__())
    
    def __getstate__(self):
        d = {}
        d['treated'] = np.asarray(self.treated)
        d['scores'] = np.asarray(self.scores)
        d['level_ids'] = np.asarray(self.level_ids)
        d['bkt_ids'] = np.asarray(self.bkt_ids)

        return d

    def __setstate__(self, d):
        self.treated = np.asarray(d['treated'])
        self.scores = np.asarray(d['scores'])
        self.level_ids = np.asarray(d['level_ids'])
        self.bkt_ids = np.asarray(d['bkt_ids'])

    cdef int init(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:] sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""

        # Initialize fields (original)
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        # Parameters
        cdef SIZE_t n_samples = self.n_samples
        cdef int n_levels = self.n_levels
        cdef int n_buckets = self.n_buckets 
        cdef int n_bkt_per_level = self.n_bkt_per_level

        # Others
        cdef SIZE_t i
        cdef SIZE_t p
        cdef int k
        cdef int b_i
        cdef bint t_i
        cdef DOUBLE_t y_i
        cdef double score_i
        cdef int b_start
        cdef int b_end

        if (start != 0) or (end != self.n_samples):
            return 0

        # Set bucket values
        for j in range(n_buckets):
            self.buckets[j].n = 0
            self.buckets[j].nt = 0
            self.buckets[j].nu = 0
            self.buckets[j].sum_yt = 0
            self.buckets[j].sum_yu = 0
            self.buckets[j].score = 0

        for p in range(start, end):
            i = samples[p]
            t_i = self.treated[i]
            y_i = self.y[i, 0]
            score_i = self.scores[i]
            b_i = self.bkt_ids[i]
      
            self.buckets[b_i].score += score_i
            self.buckets[b_i].n += 1

            if t_i == 1:
                self.buckets[b_i].nt += 1
                self.buckets[b_i].sum_yt += y_i
            else:
                self.buckets[b_i].nu += 1
                self.buckets[b_i].sum_yu += y_i

            self.weighted_n_node_samples += 1.0

        # Set level values
        for k in range(n_levels):
            self.levels[k].nt = 0
            self.levels[k].nu = 0
            self.levels[k].frac = 0
            self.levels[k].sum_yt = 0
            self.levels[k].sum_yu = 0

        for k in range(n_levels):
            b_start = k * n_bkt_per_level
            b_end = b_start + n_bkt_per_level
            for j in range(b_start, b_end):
                self.buckets[j].score /= self.buckets[j].n
                self.levels[k].nt += self.buckets[j].nt
                self.levels[k].nu += self.buckets[j].nu
                self.levels[k].sum_yt += self.buckets[j].sum_yt
                self.levels[k].sum_yu += self.buckets[j].sum_yu

        for k in range(1, n_levels):
            self.levels[k].nt += self.levels[k-1].nt
            self.levels[k].nu += self.levels[k-1].nu
            self.levels[k].sum_yt += self.levels[k-1].sum_yt
            self.levels[k].sum_yu += self.levels[k-1].sum_yu

        for k in range(n_levels):
            self.levels[k].frac = (<double> (self.levels[k].nt + self.levels[k].nu)) / n_samples

        # Reset to pos=start
        self.reset()

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef int k
        cdef int j
        cdef int n_levels = self.n_levels
        cdef int n_bkt_per_level = self.n_bkt_per_level
        cdef int n_buckets = self.n_buckets

        cdef BucketStruct[:] buckets_left = self.buckets_left
        cdef BucketStruct[:] buckets_right = self.buckets_right
        cdef int[:] n_bkt_left_level = self.n_bkt_left_level
        cdef int[:] n_bkt_right_level = self.n_bkt_right_level

        for j in range(n_buckets):
            buckets_left[j].n = 0
            buckets_left[j].nt = 0
            buckets_left[j].nu = 0
            buckets_left[j].sum_yt = 0
            buckets_left[j].sum_yu = 0
            buckets_left[j].score = 0
            buckets_right[j].n = self.buckets[j].n
            buckets_right[j].nt = self.buckets[j].nt
            buckets_right[j].nu = self.buckets[j].nu
            buckets_right[j].sum_yt = self.buckets[j].sum_yt
            buckets_right[j].sum_yu = self.buckets[j].sum_yu
            buckets_right[j].score = self.buckets[j].score

        for k in range(n_levels):
            n_bkt_left_level[k] = 0
            n_bkt_right_level[k] = n_bkt_per_level

        # Original
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = self.start

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
        cdef int k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        cdef unsigned char t_i
        cdef double y_i
        cdef double score_i
        cdef int lv_i
        cdef int b_i

        cdef BucketStruct[:] buckets_left = self.buckets_left
        cdef BucketStruct[:] buckets_right = self.buckets_right

        cdef int[:] n_bkt_left_level = self.n_bkt_left_level
        cdef int[:] n_bkt_right_level = self.n_bkt_right_level
        
        for p in range(pos, new_pos):
            i = self.samples[p]
            t_i = self.treated[i]
            y_i = self.y[i, 0]
            score_i = self.scores[i]
            b_i = self.bkt_ids[i]
            lv_i = self.level_ids[i]

            # Update buckets left
            buckets_left[b_i].score = (buckets_left[b_i].score * buckets_left[b_i].n + score_i)\
                                        / (buckets_left[b_i].n + 1) 
            buckets_left[b_i].n += 1
            if t_i == 1:
                buckets_left[b_i].nt += 1
                buckets_left[b_i].sum_yt += y_i
            else:
                buckets_left[b_i].nu += 1
                buckets_left[b_i].sum_yu += y_i

            # Update bucktes right
            buckets_right[b_i].score = (buckets_right[b_i].score * buckets_right[b_i].n - score_i)\
                                        / max(1, buckets_right[b_i].n - 1) 
            buckets_right[b_i].n -= 1
            if t_i == 1:
                buckets_right[b_i].nt -= 1
                buckets_right[b_i].sum_yt -= y_i
            else:
                buckets_right[b_i].nu -= 1
                buckets_right[b_i].sum_yu -= y_i

            # Update number of buckets at this level
            if buckets_left[b_i].n == 1:
                n_bkt_left_level[lv_i] += 1
            if buckets_right[b_i].n == 0:
                n_bkt_right_level[lv_i] -= 1

            self.weighted_n_left += 1.0

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = new_pos

        return 0

    cdef void reset_to_no_swap(self, bint fix_left) nogil:

        cdef int n_levels = self.n_levels
        cdef int n_bkt_per_level = self.n_bkt_per_level
        cdef int k

        self.fix_left = fix_left

        if fix_left:
            self.buckets_fxd = self.buckets_left
            self.buckets_sft = self.buckets_right
        else:
            self.buckets_fxd = self.buckets_right
            self.buckets_sft = self.buckets_left

        for k in range(n_levels):
            if fix_left: 
                self.n_bkt_fxd_level[k] = self.n_bkt_left_level[k]
                self.n_bkt_sft_level[k] = self.n_bkt_right_level[k]
            else:
                self.n_bkt_fxd_level[k] = self.n_bkt_right_level[k]
                self.n_bkt_sft_level[k] = self.n_bkt_left_level[k]

            self.first_sft_level[k] = n_bkt_per_level * k
            self.last_fxd_level[k] = self.first_sft_level[k] + n_bkt_per_level - 1

            self.curr_levels[k].nt = self.levels[k].nt
            self.curr_levels[k].nu = self.levels[k].nu
            self.curr_levels[k].frac = self.levels[k].frac
            self.curr_levels[k].sum_yt = self.levels[k].sum_yt
            self.curr_levels[k].sum_yu = self.levels[k].sum_yu

 
    cdef ShiftRecord create_shift_record(self, int level_h) nogil:
        
        cdef int n_levels = self.n_levels
        cdef int[:] n_bkt_fxd_level = self.n_bkt_fxd_level
        cdef int[:] n_bkt_sft_level = self.n_bkt_sft_level
        cdef int[:] last_fxd_level = self.last_fxd_level
        cdef int[:] first_sft_level = self.first_sft_level
        cdef BucketStruct[:] buckets_fxd = self.buckets_fxd
        cdef BucketStruct[:] buckets_sft = self.buckets_sft

        cdef ShiftRecord sr
        cdef int b_h
        cdef int b_l
        cdef int ix
        cdef double shift
        cdef int level_l = level_h + 1

        cdef int k

        if  n_bkt_fxd_level[level_h] > 0 and n_bkt_sft_level[level_l] > 0:

            b_h = last_fxd_level[level_h]
            while buckets_fxd[b_h].n == 0:
                b_h -= 1

            b_l = first_sft_level[level_l]
            while buckets_sft[b_l].n == 0:
                b_l += 1

            shift = buckets_fxd[b_h].score - buckets_sft[b_l].score \
                    + SCORE_THRESHOLD
            
            sr.b_h = b_h
            sr.b_l = b_l
            sr.level_h = level_h
            sr.level_l = level_l
            sr.shift = shift

            _add_to_shift_queue(sr, self.shift_queue)


    cdef void expand_shift_queue(self, int level_h, int level_l) nogil:

        cdef int n_levels = self.n_levels

        self.create_shift_record(level_h)

        if (level_h >= 1) and self.n_bkt_sft_level[level_h] == 1:
            self.create_shift_record(level_h-1)

        if (level_l + 1 < n_levels) and self.n_bkt_fxd_level[level_l] == 1:
            self.create_shift_record(level_l)

    cdef double ate_at_k(self, int k) nogil:

        cdef int nt, nu
        cdef double sum_yt, sum_yu
        cdef double ate

        nt = self.curr_levels[k].nt
        nu = self.curr_levels[k].nu
        sum_yt = self.curr_levels[k].sum_yt
        sum_yu = self.curr_levels[k].sum_yu

        ate = sum_yt / max(1, nt) - sum_yu / max(1, nu)

        return ate

    cdef double auuc_increment(self, int b_i, int b_j, int level_h) nogil:

        cdef double ate_before
        cdef double ate_after
        cdef double auuc_incr
        cdef BucketStruct[:] buckets_fxd = self.buckets_fxd
        cdef BucketStruct[:] buckets_sft = self.buckets_sft
        cdef LevelStruct[:] curr_levels = self.curr_levels

        # AUUC increment
        ate_before = self.ate_at_k(level_h)

        curr_levels[level_h].nt += buckets_sft[b_j].nt - buckets_fxd[b_i].nt
        curr_levels[level_h].nu += buckets_sft[b_j].nu - buckets_fxd[b_i].nu
        curr_levels[level_h].sum_yt += buckets_sft[b_j].sum_yt - buckets_fxd[b_i].sum_yt
        curr_levels[level_h].sum_yu += buckets_sft[b_j].sum_yu - buckets_fxd[b_i].sum_yu

        ate_after = self.ate_at_k(level_h)
        auuc_incr = curr_levels[level_h].frac * (ate_after - ate_before)

        return auuc_incr

    cdef (double, double) positive_shift_and_auuc_increment(self, bool fix_left) nogil:
        
        cdef int n_levels = self.n_levels
        cdef double curr_shift = .0
        cdef double best_shift = .0
        cdef double curr_auuc_incr = .0
        cdef double best_auuc_incr = .0

        cdef int b_i
        cdef int b_j
        cdef int level_h
        cdef int level_l
        cdef ShiftRecord sr
        cdef int k

        self.reset_to_no_swap(fix_left)

        # initialize shift queue
        for k in range(n_levels-1):
            self.create_shift_record(k)

        while not self.shift_queue.empty():

            pop_heap(self.shift_queue.begin(), self.shift_queue.end(), &_compare_records)
            sr = self.shift_queue.back()
            self.shift_queue.pop_back()

            # printf("curr_shift = %lf\n", sr.shift)

            # Swap the levels of i and j
            b_i, b_j = sr.b_h, sr.b_l
            level_h, level_l = sr.level_h, sr.level_l

            # Update the AUUC increment
            curr_shift = sr.shift
            curr_auuc_incr += self.auuc_increment(b_i, b_j, level_h)
            if curr_auuc_incr > best_auuc_incr:
                best_shift = curr_shift
                best_auuc_incr = curr_auuc_incr

            # Update statistics
            self.n_bkt_fxd_level[level_h] -= 1
            self.n_bkt_fxd_level[level_l] += 1
            self.n_bkt_sft_level[level_h] += 1
            self.n_bkt_sft_level[level_l] -= 1

            self.last_fxd_level[level_h] = b_i - 1
            self.first_sft_level[level_l] = b_j + 1

            # Expand the shift queue
            self.expand_shift_queue(level_h, level_l)
        
        return best_shift, best_auuc_incr

    cdef void shift_and_auuc_increment(self, StumpStruct* stump) nogil:

        cdef double shift_fl
        cdef double shift_fr
        cdef double auuc_incr_fl
        cdef double auuc_incr_fr

        stump[0].pos = self.pos
        shift_fl, auuc_incr_fl = self.positive_shift_and_auuc_increment(fix_left=True) 
        shift_fr, auuc_incr_fr = self.positive_shift_and_auuc_increment(fix_left=False) 
        if auuc_incr_fl >= auuc_incr_fr:
            stump[0].shift_left = .0
            stump[0].shift_right = shift_fl
            stump[0].auuc_incr = auuc_incr_fl
        else:
            stump[0].shift_left = shift_fr
            stump[0].shift_right = .0
            stump[0].auuc_incr = auuc_incr_fr

    cdef double node_impurity(self) nogil:
        return 1.0

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        impurity_left[0] = 1.0
        impurity_right[0] = 1.0
        
    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        if self.stump.pos == 0:     # Parent node
            dest[0] = 0
        elif self.stump.pos == self.end:    # Left node
            dest[0] = self.stump.shift_left
        elif self.stump.pos == self.start:   # Right node
            dest[0] = self.stump.shift_right

    cdef double proxy_impurity_improvement(self) nogil:

        cdef StumpStruct curr_stump
        cdef double improvement

        self.shift_and_auuc_increment(&curr_stump)
        improvement = curr_stump.auuc_incr
        if improvement > self.stump.auuc_incr:
            self.stump = curr_stump

        return max(0, improvement)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:

        return self.stump.auuc_incr


    def set_sample_parameters(self, unsigned char[:] treated, double[:] scores, int[:] level_ids, 
                            int[:] bkt_ids):

        self.treated = treated
        self.scores = scores 
        self.level_ids = level_ids
        self.bkt_ids = bkt_ids







