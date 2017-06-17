# std
import bisect
from collections import defaultdict, namedtuple
import heapq

# 3p
import numpy as np


## Function to learn and plot piecewise regressions.


def piecewise(t, v, min_stop_frac=0.03):
    """ Fits a piecewise (aka "segmented") regression.
    Params:
        t (listlike of ints or floats): independent/predictor variable values
        v (listlike of ints or floats): dependent/outcome variable values
        min_stop_frac (float between 0 and 1): the fraction of total error that
            a merge must account for to be considered big enough to stop merging;
            the default is usually adequate, but this may be increased to make
            merging more aggressive (leading to fewer segments in the result)
    Returns:
        A FittedModel object that can be used for interpolation and extrapolation.
    """
    # Validate the inputs, and force t and v to be np.arrays sorted in
    # ascending t order.
    t, v = _preprocess(t, v)

    # Initialize the segments.
    init_segments = _get_initial_segments(t, v)
    seg_tracker = SegmentTracker(init_segments)

    merges = _get_initial_merges(t, v, seg_tracker.segments)
    # Use a min heap to track potential merges. At the top of the heap
    # will be the next best merge (smallest increase in error).
    heapq.heapify(merges)

    # Greedily make the next best merge until we've merged everything together into
    # one segment.
    cum_cost, biggest_cost_increase = 0.0, 0.0
    best_segments = seg_tracker.segments
    while len(seg_tracker) > 1:
        # Identify the next merge to be executed.
        next_merge = _get_next_merge(merges, seg_tracker)

        # If the next merge increases the error by a larger amount than any
        # merge so far, remember the current state (which might end up being the
        # "best"). To prevent stopping too early (for example, in cases where
        # there should only be one segment), use min_stop_frac to keep on
        # remembering the current state as the best state if no single
        # merge has accounted for a significant part of the total error.
        cum_cost += next_merge.cost
        cost_increase = next_merge.cost - biggest_cost_increase
        biggest_cost_increase = max(biggest_cost_increase, cost_increase)
        if biggest_cost_increase < min_stop_frac*cum_cost or \
                cost_increase == biggest_cost_increase:
            best_segments = seg_tracker.segments

        # Execute the next merge.
        # Update segments, replacing the two old ones with the one new one.
        seg_tracker.replace(next_merge.left_seg, next_merge.right_seg, next_merge.new_seg)
        # Add new potential merges.
        neighbors = seg_tracker.get_neighbors(next_merge.new_seg)
        for neighbor in neighbors:
            left_seg, right_seg = sorted([next_merge.new_seg, neighbor])
            heapq.heappush(merges, _make_merge(t, v, left_seg, right_seg))

    if biggest_cost_increase < min_stop_frac*cum_cost:
        # This path is needed for the case where there is only one segment, because
        # best_segments isn't updated after merging in the loop above.
        best_segments = seg_tracker.segments

    fitted_segments = [
        FittedSegment(t[seg.start_index], t[min(seg.end_index, len(t)-1)], seg.coeffs)
        for seg in best_segments
    ]
    return FittedModel(fitted_segments)


## Data structures used for representing the fitted model returned by `piecewise()`.


class FittedSegment(namedtuple('FittedSegment',
    [
        'start_t',  # (float) first t value to which this segment applies
        'end_t',    # (float) first t value to which this segment no longer applies
        'coeffs'    # (tuple of floats) regression coefficients
    ]
)):
    def predict(self, t_new):
        return _predict(self.coeffs, t_new)


class FittedModel(object):
    """ Completely defines the result of a piecewise regression.
    The `segments` attribute contains a list of FittedSegments.
    """

    def __init__(self, fitted_segments):
        self.segments = fitted_segments
        self._starts = [fs.start_t for fs in fitted_segments]

    def __repr__(self):
        return 'FittedModel with segments:\n' + '\n'.join(
            ['* ' + seg.__repr__() for seg in self.segments]
        )

    def predict(self, t_new):
        """ Use the segments in this model to predict the v value for new t values.
        Params:
            t_new (np.array): t values for which predictions should be made
        Returns:
            np.array of predictions
        """
        v_hats = np.empty_like(t_new, dtype=float)
        for idx, t in enumerate(t_new):
            # Find the applicable segment.
            seg_index = bisect.bisect_left(self._starts, t) - 1
            seg = self.segments[max(0, seg_index)]
            # Use it for prediction
            v_hats[idx] = seg.predict(t)
        return v_hats


## Data structures used during the fitting of the regression in `piecewise()`.


# Segment represents a time range and a linear regression fit through it.
Segment = namedtuple('Segment',
    [
        'start_index',  # (int) zero-based index of start time
        'end_index',    # (int) zero-based index of non-inclusive end time
        'coeffs',       # (tuple of floats) regression coefficients
        'error'         # (float) the total error in the segment
    ]
)


# Merge represents a potential merge of two neighboring segments.
Merge = namedtuple('Merge',
    [
        'cost',       # (float) increase in sum of squared error that would result from executing this merge
        'left_seg',   # (Segment)
        'right_seg',  # (Segment)
        'new_seg'     # (Segment) the Segment that would result from merging combining left_seg and right_seg
    ]
)


class SegmentTracker(object):
    """ Utility class for tracking the state of the piecewise regression (i.e.,
    what are the current segments based on the set of merges that have been
    executed so far).
    """

    def __init__(self, segments):
        self._ordered_segments = sorted(segments)
        self._segment_set = set(segments)

    def __len__(self):
        return len(self._ordered_segments)

    def contains(self, segment):
        """ Returns True if segment is currently valid; False otherwise. """
        return segment in self._segment_set

    def get_neighbors(self, segment):
        """ Returns a list of Segments, containing the 0, 1, or 2 segments
        adjacent to the given Segment.
        """
        index = bisect.bisect_left(self._ordered_segments, segment)
        neighbors = []
        if index - 1 >= 0:
            neighbors.append(self._ordered_segments[index-1])
        if index + 1 < len(self._ordered_segments):
            neighbors.append(self._ordered_segments[index+1])
        return neighbors

    def replace(self, old_left_segment, old_right_segment, new_segment):
        """ Insert a new segment and remove the two existing segments
        from which it was created.
        """
        # Update the list of Segments.
        left_index = bisect.bisect_left(self._ordered_segments, old_left_segment)
        self._ordered_segments[left_index] = new_segment
        del self._ordered_segments[left_index+1]

        # Update the set of Segments
        self._segment_set.remove(old_left_segment)
        self._segment_set.remove(old_right_segment)
        self._segment_set.add(new_segment)

    @property
    def segments(self):
        return tuple(self._ordered_segments)


## Helper functions for doing piecewise regression.


def _preprocess(t, v):
    """ Raises and exception if any of the inputs are not valid.
    Otherwise, returns a list of Points, ordered by t.
    """
    # Validate the inputs.
    if len(t) != len(v):
        raise ValueError('`t` and `v` must have the same length.')
    t_arr, v_arr = np.array(t), np.array(v)
    if not np.all(np.isfinite(t)):
        raise ValueError('All values in `t` must be finite.')
    finite_mask = np.isfinite(v_arr)
    if np.sum(finite_mask) < 2:
        raise ValueError('`v` must have at least 2 finite values.')
    t_arr, v_arr = t_arr[finite_mask], v_arr[finite_mask]
    if len(np.unique(t_arr)) != len(t_arr):
        raise ValueError('All `t` values must be unique.')

    # Order both arrays by t-values.
    sort_order = np.argsort(t_arr)
    t_arr, v_arr = t_arr[sort_order], v_arr[sort_order]

    return t_arr, v_arr


def _get_initial_segments(t, v):
    """ Returns a list of Segments. Each Segment is of length 1, 2, or 3. They
    are created by using even-indexed points as seeds and attaching odd-indexed
    points to the neighboring seed with the closer v value.
    This initialization procedure exists to decrease the odds of bad initial
    merges. If initial segments were each a single point, then merging any two
    neighboring points would be equally attractive to our algorithm, because the
    squared error of a line fit through any pair of points is zero. However,
    in the case that the data looks like [1, 1, 1, 1, 10, 10, 10, 10], we would
    prefer to avoid the 1 and neighboring 10 from starting out in the same
    segment. This initialization does this by doing initial merges based on
    absolute difference rather than regression error. Unfortunately, there can
    still be suboptimal initializations, as in this case, where the two 1s will
    be initialized in the same segment: [19, 10, 1, 1, -8, -17]
    """
    seed_assignments = defaultdict(list)
    for i in range(1, len(t), 2):
        left_diff = abs(v[i-1] - v[i])
        right_diff = abs(v[i+1] - v[i]) if len(v) > i+1 else np.inf
        best_seed = i-1 if left_diff < right_diff else i+1
        seed_assignments[best_seed].append(i)
    segments = []
    for i in range(0, len(t), 2):
        indices = seed_assignments[i] + [i]
        start_index, end_index = min(indices), max(indices)+1
        segments.append(_make_segment(t, v, start_index, end_index))
    return segments


def _get_initial_merges(t, v, segments):
    """ Returns a list of all possible Merges for the given list of Segments.
    """
    return [
        _make_merge(t, v, segments[i], segments[i+1])
        for i in range(len(segments)-1)
    ]


def _get_next_merge(merges, segment_tracker):
    """ Returns the valid Merge that has the lowest cost.
    Params:
        merges: a heapified list of Merges
        segment_tracker: a SegmentTracker with the currently valid segments;
            any Merge referencing a Segment not in the tracker is no longer valid
    """
    while True:
        next_merge = heapq.heappop(merges)
        if (segment_tracker.contains(next_merge.left_seg) and
                segment_tracker.contains(next_merge.right_seg)):
            return next_merge


def _make_segment(t, v, start_index, end_index):
    """ Returns a Segment that starts at start_index and ends at
    the non-inclusive end_index.
    """
    coeffs, error = _fit_line(t, v, start_index, end_index)
    return Segment(start_index, end_index, coeffs, error)


def _make_merge(t, v, left_seg, right_seg):
    """ Returns a Merge combining the left_seg and right_seg Segments.
    """
    new_seg = _make_segment(t, v, left_seg.start_index, right_seg.end_index)
    cost = new_seg.error - left_seg.error - right_seg.error
    return Merge(cost, left_seg, right_seg, new_seg)


def _fit_line(t, v, start_index, end_index):
    """ Fits and OLS regression for the set of t and v values in the given index
    range. Returns (coefficients of line, sum of squared error).
    """
    t_slice = t[start_index:end_index]
    v_slice = v[start_index:end_index]
    A = np.array([np.ones_like(t_slice), t_slice])
    coeffs, error = np.linalg.lstsq(A.T, v_slice)[0:2]
    return tuple(coeffs), 0.0 if len(error) == 0 else float(error)


def _predict(coeffs, t):
    """ Given OLS coefficients, predict the corresponding v values for the given
    t values.
    """
    return np.sum(np.array(coeffs) * np.array([1.0, t]))
