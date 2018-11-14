import collections
import itertools
from collections import OrderedDict
from typing import DefaultDict

import cv2
import numpy as np
from scipy.spatial import distance as dist
from tracker.track import Track
from tracker.track_merger import TrackMerger
import logging

logger = logging.getLogger('Tracker')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class DetectionBasedTracker:
    def __init__(self, tracker_type='CSRT', merge_interval=40, merge_max_perc=0.4, max_disappear=20,
                 iou_thresh=0.6):
        self.id_to_trackers = OrderedDict()
        self.disappeared = OrderedDict()
        self.tracker_type = tracker_type
        self.cntr = 0
        self.merge_interval = merge_interval
        self.merge_max_perc = merge_max_perc
        self.max_disappear = max_disappear
        self.iou_thresh = iou_thresh
        self.track_merger = TrackMerger(merge_interval, merge_max_perc)
        self.last_deleted = []
        self.last_registered = []

    def create_tracker(self):
        tracker_type = self.tracker_type
        if tracker_type == 'BOOSTING':
            return cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            return cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            raise Exception('incorrect tracker')


    def update(self, frame, bbs, frame_cnt):
        """
        :param frame: img
        :param bbs: [[x,y,w,h]]
        :param frame_cnt: current frame number
        :return: list of trackers that are corresp. to bbs
        """
        self.last_deleted = []
        self.last_registered = []

        tracks_bbs = []
        track_bbs_to_track = {}
        track_bbs_cntr = 0
        for id, track in self.id_to_trackers.copy().items():
            ok, bbox = track.tracker.update(frame)
            if not ok:
                #logger.error('not ok')
                self.disappeared[id] += 1
                if self.disappeared[id] > self.max_disappear:
                    self.deregister(id)
            else:
                #logger.debug('OK')
                track.last_tracker_bb = bbox
                tracks_bbs.append(bbox)
                track_bbs_to_track[track_bbs_cntr] = track
                track_bbs_cntr += 1


        if len(tracks_bbs) == 0 or len(bbs) == 0:
            res = []
            for i in range(0, len(bbs)):
                res.append([self.register(frame, bbs[i])])
                res[i][0].path.append(bbs[i])

            return res
        else:
            res_bb_to_track_dict = collections.defaultdict(list)

            D = 1. - dist.cdist(np.asarray(tracks_bbs), np.array(bbs), metric=self.bb_intersection_over_union)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                if D[row, col] > 1 - self.iou_thresh:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                id = track_bbs_to_track[row].id
                self.id_to_trackers[id].path.append(bbs[col])
                self.disappeared[id] = 0
                res_bb_to_track_dict[col].append(self.id_to_trackers[id])

                # indicate that we have examined each of the row and
                # column indexes, respectively
                # usedRows.add(row)
                # we can use multiple tracks for one source bbox
                usedCols.add(col)
                usedRows.add(row)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)


            # loop over the unused row indexes
            for row in unusedRows:
                id = track_bbs_to_track[row].id
                self.disappeared[id] += 1

                if self.disappeared[id] > self.max_disappear:
                    self.deregister(id)

            for col in unusedCols:
                res_bb_to_track_dict[col].append(self.register(frame, bbs[col]))

            assert len(res_bb_to_track_dict) == len(bbs)
            res_bb_to_track_dict = OrderedDict(sorted(res_bb_to_track_dict.items()))
            self.merge_trackers(res_bb_to_track_dict, frame_cnt)

            # update pathes
            for i, tracks in res_bb_to_track_dict.items():
                for track in tracks:
                    track.path.append(bbs[i])

            assert len(res_bb_to_track_dict) == len(bbs)
            return res_bb_to_track_dict.values()

    def merge_trackers(self, res_dict, frame_cnt):
        for i, vals in res_dict.items():
            if len(vals) > 0:
                res_dict[i] = self.track_merger.merge_count(vals, frame_cnt, self.deregister)

    def register(self, frame, bb):
        tracker = self.create_tracker()
        tracker.init(frame, tuple(bb))
        res = self.id_to_trackers[self.cntr] = Track(self.cntr, tracker)
        res.last_tracker_bb = bb
        self.disappeared[self.cntr] = 0
        self.last_registered.append(res)
        self.cntr += 1
        return res

    def deregister(self, object_id):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        self.id_to_trackers[object_id].is_active = False
        self.last_deleted.append(self.id_to_trackers[object_id])
        del self.id_to_trackers[object_id]
        del self.disappeared[object_id]

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        boxA = boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]
        boxB = boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
