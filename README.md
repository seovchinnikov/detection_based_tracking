# detection_based_tracking
Tracking under assumption that tracks are originated from detections. 

When trackers' results diverge from detections new tracker is created but if it ties up with some of other trackers later it will be merged with its dublicates.

If some tracker is not matched with any detections for a long period removal process will be started
