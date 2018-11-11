# detection_based_tracking
Tracking under assumption that tracks are originated from detections. 

When trackers' results diverge from detections new tracker is created but if it ties up with some of other trackers later it will be merged with its dublicates. So one detection can be matched with multiple trackers but it is only matter of time to merge them (so it can be used in situations where we need to maintain statistics of tracks during long period compared to merge period). Also this method is resistant to low recall detectors.

If some tracker is not matched with any detections for a long period removal process will be started
