class Track:
    def __init__(self, id, tracker):
        self.id = id
        self.tracker = tracker
        self.last_tracker_bb = None
        self.path = []
        self.is_active = True
        self.is_merged = False

