"""Per-camera tracking (docs/05): detect -> ByteTrack -> tracklets + attributes."""
from .attributes import AttributeProvider, NoOpAttributes, crop_of
from .run import (
    iter_image_frames, iter_video_frames, render_frame, summarize,
    track_sequence, track_stream,
)
from .jersey import JerseyReader, NumberLocalizer, read_jerseys
from .reid import OSNetEmbedder, track_embeddings
from .teams import SiglipEmbedder, assign_teams
from .tracker import DEFAULT_TRACK_CLASSES, ByteTrackTracker
from .types import CLASS_NAMES, Track

__all__ = [
    "Track", "CLASS_NAMES", "ByteTrackTracker", "DEFAULT_TRACK_CLASSES",
    "track_sequence", "track_stream", "render_frame", "iter_video_frames",
    "iter_image_frames", "summarize", "AttributeProvider", "NoOpAttributes", "crop_of",
    "SiglipEmbedder", "assign_teams", "OSNetEmbedder", "track_embeddings",
    "JerseyReader", "NumberLocalizer", "read_jerseys",
]
