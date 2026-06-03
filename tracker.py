from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2):
        """
        Initializes the DeepSORT tracker.
        
        Args:
            max_age: Maximum number of missed frames before a track is deleted.
            n_init: Number of consecutive detections before the track is confirmed.
            nms_max_overlap: Maximum non-max suppression overlap.
            max_cosine_distance: Threshold for cosine distance metric (visual re-id matching).
        """
        self.max_age = max_age
        self.n_init = n_init
        self.nms_max_overlap = nms_max_overlap
        self.max_cosine_distance = max_cosine_distance
        
        # Check if PyTorch GPU is available for the DeepSORT embedder
        self.embedder_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                self.embedder_gpu = True
        except ImportError:
            pass

        # We configure the DeepSORT tracker
        self.embedder_name = "mobilenet"
        self.half = True if self.embedder_gpu else False # half precision only on GPU
        self.bgr = True
        self.nn_budget = None
        
        self.reset()

    def reset(self):
        """
        Resets and re-initializes the DeepSORT tracker state.
        This is useful when switching input sources or restarting the stream.
        """
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            nms_max_overlap=self.nms_max_overlap,
            max_cosine_distance=self.max_cosine_distance,
            nn_budget=self.nn_budget,
            embedder=self.embedder_name,
            half=self.half,
            bgr=self.bgr,
            embedder_gpu=self.embedder_gpu
        )

    def update(self, detections, frame):
        """
        Updates the tracker with new detections and the current frame.
        
        Args:
            detections: List of tuples ( [x, y, w, h], confidence, class_name )
            frame: Raw BGR frame image.
            
        Returns:
            active_tracks: List of dictionaries with keys:
                           "track_id": string ID,
                           "bbox": [left, top, right, bottom],
                           "class_name": string name,
                           "confidence": float detection confidence
        """
        # Run tracker update
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []
        
        for track in tracks:
            # Skip unconfirmed tracks (must see target in n_init consecutive frames to confirm)
            if not track.is_confirmed():
                continue
                
            # Skip tracks that were not updated in the current frame
            if track.time_since_update > 1:
                continue
                
            # Retrieve bounding box coordinates in LTRB format [left, top, right, bottom]
            ltrb = track.to_ltrb()
            
            # Retrieve track properties
            track_id = track.track_id
            class_name = track.get_class()
            
            # Retrieve detection confidence
            det_conf = getattr(track, 'det_conf', None)
            if det_conf is None:
                det_conf = 1.0
                
            # Ensure coordinates are within image boundaries
            left, top, right, bottom = map(int, ltrb)
            
            active_tracks.append({
                "track_id": str(track_id),
                "bbox": [left, top, right, bottom],
                "class_name": class_name,
                "confidence": float(det_conf)
            })
            
        return active_tracks
