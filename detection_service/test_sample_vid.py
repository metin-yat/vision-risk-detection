import cv2
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import time

MODEL_ID = "construction-safety-gsnvb/1"
VIDEO_PATH = "C:\\Users\\metin\\Desktop\\minen\\vision-risk-detection\\seg_service\\source\\true1.mp4"
API_KEY = "DgLdimkc0g92ivlm5aMm"
CONFIDENCE_THRESHOLD = 0.1
TARGET_SIZE = (640, 640)

total_frames = 0
frames_with_human = 0
frames_with_helmet = 0
prev_time = 0

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def render_callback(predictions: dict, video_frame: VideoFrame):
    global total_frames, frames_with_human, frames_with_helmet, prev_time

    resized_frame = cv2.resize(video_frame.image, TARGET_SIZE)
    detections = sv.Detections.from_inference(predictions)
    
    h, w = video_frame.image.shape[:2]
    detections.xyxy[:, [0, 2]] *= TARGET_SIZE[0] / w
    detections.xyxy[:, [1, 3]] *= TARGET_SIZE[1] / h

    total_frames += 1
    class_names = detections.data.get('class_name', [])
    
    has_human = 'person' in class_names
    has_helmet = 'helmet' in class_names

    if has_human:
        frames_with_human += 1
    if has_helmet:
        frames_with_helmet += 1

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    annotated_frame = box_annotator.annotate(scene=resized_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    h_color = (0, 255, 0) if has_human else (0, 0, 255)
    k_color = (0, 255, 0) if has_helmet else (0, 0, 255)
    
    cv2.putText(annotated_frame, f"HUMAN: {'OK' if has_human else 'NOT FOUND'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)
    cv2.putText(annotated_frame, f"HELMET: {'OK' if has_helmet else 'NOT FOUND'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, k_color, 2)

    cv2.imshow("Critical Check: Human & Helmet", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.terminate()

pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    video_reference=VIDEO_PATH,
    on_prediction=render_callback,
    api_key=API_KEY,
    confidence=CONFIDENCE_THRESHOLD
)

print("Processing started...")
pipeline.start()
pipeline.join()

cv2.destroyAllWindows()
if total_frames > 0:
    human_rate = (frames_with_human / total_frames) * 100
    helmet_rate = (frames_with_helmet / total_frames) * 100
    
    print("\n" + "="*45)
    print(f"{'MODAL CRITICAL SUCCESS ANALYSIS':^45}")
    print("="*45)
    print(f"Total Frame Count      : {total_frames}")
    print("-" * 45)
    print(f"HUMAN DETECTION")
    print(f"  - Detected Frames        : {frames_with_human}")
    print(f"  - Success Rate         : %{human_rate:.2f}")
    print("-" * 45)
    print(f"HELMET DETECTION")
    print(f"  - Detected Frames        : {frames_with_helmet}")
    print(f"  - Success Rate         : %{helmet_rate:.2f}")
    print("="*45)
else:
    print("Video could not be processed.")