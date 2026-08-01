# Image Utils

> Image processing utilities: crop, resize, blur/brightness scoring, JPEG encoding, Cloudinary upload, annotation drawing.

**File**: `utils/image_utils.py`

## Key functions

### `crop_person(frame, box)`
Crops person bounding box from frame. Returns numpy array.

### `compute_blur_score(face_crop)`
```python
gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if 3-channel
return cv2.Laplacian(gray, cv2.CV_64F).var()
```
Higher = sharper. Validity gate: ≥ 40.

### `compute_brightness(face_crop)`
```python
hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
return np.mean(hsv[:, :, 2])  # V channel
```
Range 0-255. Validity gate: [35, 255].

### `draw_annotations(frame, all_tracks)`
Draws bounding boxes, track IDs, names, and status labels on frame.

### `save_image(array, path)`
Saves numpy array as JPEG file.

### `resolve_track_image_url(track)`
Returns Cloudinary URL if available, else local path.

### `upload_to_cloudinary(image_path)` / `upload_jpeg_to_cloudinary(jpeg_bytes)`
Uploads image to Cloudinary. Returns URL. Falls back to local capture on failure.

### `compute_iou(box1, box2)`
Intersection over Union for bounding boxes. Used for duplicate detection.

## See also
- [[Quality Assessment]] — how blur/brightness are used
- [[Face Detection & Embedding]] — crop_person usage
- [[Track State]] — draw_annotations called from camera loop
