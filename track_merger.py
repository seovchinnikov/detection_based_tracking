import itertools
import time

import math
from collections import defaultdict


class TrackMerger:

    def __init__(self, interval=30, limit=0.3):
        self.interval = interval
        self.cells_per_interval = int(math.ceil(float(interval)))
        self.counters = []
        self.nullify_cntrs()
        self.limit = limit
        self.counter = defaultdict(int)

    def merge_count(self, tracks, frames_now, dereg_func):
        if len(tracks) <= 1:
            return tracks

        tracks = sorted(tracks, key=lambda x: x.id)
        pairs = list(itertools.combinations(tracks, 2))
        cur_time = frames_now
        bucket_val_now = int(cur_time)
        bucket_now = bucket_val_now % self.cells_per_interval

        val_ex = self.counters[bucket_now]
        for t1, t2 in val_ex:
            self.counter[str(t1.id) + '$' + str(t2.id)] -= 1
            if self.counter[str(t1.id) + '$' + str(t2.id)] == 0:
                del self.counter[str(t1.id) + '$' + str(t2.id)]

        self.counters[bucket_now] = [(t1, t2) for t1, t2 in pairs]
        for t1, t2 in pairs:
            if not t1.is_active or not t2.is_active:
                continue
            self.counter[str(t1.id) + '$' + str(t2.id)] += 1
            if self.counter[str(t1.id) + '$' + str(t2.id)] >= self.interval * self.limit:
                dereg_func(t2.id)

        res = [val for val in tracks if val.is_active]
        if len(res) == 0:
            raise Exception('Shit happens')

        return res

    def nullify_cntrs(self):
        self.counters = [[]] * self.cells_per_interval
