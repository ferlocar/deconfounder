cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DTYPE_t, int 
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np

cdef struct ShiftRecord:
    int b_h
    int b_l
    int level_h
    int level_l
    double shift

cdef packed struct BucketStruct:
    int n
    int nt
    int nu
    double sum_yt
    double sum_yu
    double score

cdef packed struct LevelStruct:
    int nt
    int nu
    double frac
    double sum_yt
    double sum_yu

cdef struct StumpStruct:
    SIZE_t pos
    double shift_left
    double shift_right
    double auuc_incr
    

cdef class DeconfoundAUUC(Criterion):

    # Bucket
    cdef BucketStruct[:] buckets
    cdef BucketStruct[:] buckets_left
    cdef BucketStruct[:] buckets_right

    # Level
    cdef LevelStruct[:] levels
    cdef LevelStruct[:] curr_levels

    # Left and right
    cdef int[:] n_bkt_left_level
    cdef int[:] n_bkt_right_level

    # Fix and shift
    cdef bint fix_left
    cdef int[:] n_bkt_fxd_level
    cdef int[:] n_bkt_sft_level
    cdef int[:] last_fxd_level
    cdef int[:] first_sft_level
    cdef BucketStruct[:] buckets_fxd
    cdef BucketStruct[:] buckets_sft

    cdef vector[ShiftRecord] shift_queue

    # Data
    cdef unsigned char[:] treated
    cdef double[:] scores
    cdef int[:] bkt_ids
    cdef int[:] level_ids

    # Parameters
    cdef int n_levels
    cdef int n_buckets
    cdef int n_bkt_per_level

    cdef StumpStruct stump

    cdef void reset_to_no_swap(self, bint fix_left) nogil
    cdef ShiftRecord create_shift_record(self, int level_h) nogil
    cdef void expand_shift_queue(self, int level_h, int level_l) nogil
    cdef double ate_at_k(self, int k) nogil
    cdef double auuc_increment(self, int b_i, int b_j, int level_h) nogil
    cdef (double, double) positive_shift_and_auuc_increment(self, bint fix_left) nogil
    cdef void shift_and_auuc_increment(self, StumpStruct* stump) nogil


