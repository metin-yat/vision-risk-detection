from typing import Any


from data_loader import get_source
from config import Config
import logging, cv2, os, utils, time
from dotenv import load_dotenv
from statistics import mean
import threading, uuid, json
from datetime import datetime


load_dotenv()
api_key = os.getenv("API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]||[%(levelname)s]|(%(name)s)|: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ],
    force= True
);logger = logging.getLogger(__name__)

def main():
    logger.info("Risk Detection Service starting...")

    # loading the model from Roboflow. 
    model = utils.initialize_model(Config.MODEL_PATH,
                                            api_key)
    
    if not utils.test_model(model=model):
        logger.error("Model is not working, please check: Model related env. variables")
    else:
        logger.info("Model works fine, moving to next stages of the pipeline.")

    # Getting the source of the frames (Image sequentials, videos, camera)
    try:
        cap = get_source(Config.SOURCE_DIR)
        logger.info(f"Source successfully opened: {Config.SOURCE_DIR}")
    except Exception as e:
        logger.error(f"Failed to get source: {e}")
        return

    # For batching purposes, some processing steps are done there.
    streamer = utils.SmartStreamer(cap, batch_size=10)
    streamer.start_stream()

    no_person_counter = 0
    processed_batches_counter = 0
    event_buffer = []

    start_time = time.perf_counter()

    try:
        while streamer.is_running or len(streamer.frame_buffer) > 0:
            current_batch = streamer.get_batch()
            
            if current_batch:
                # Ensure the batch is a list of numpy arrays
                # Using image=current_batch is usually enough if the arrays are standard
                try:
                    results = model.infer(image=current_batch,
                            confidence=Config.CONFIDENCE_THRESHOLD)
                except Exception as e:
                    logger.error(f"Inference failed: {e}")
                    continue

                # Filter for human detection
                valid_frames_data = []

                for idx, res in enumerate[Any](results):
                    # Get predictions
                    predictions = getattr(res, 'predictions', [])
                    
                    # Logic: Check if 'person' class exists in this specific frame
                    has_person = any(getattr(p, 'class_name', '').lower() == 'person' for p in predictions)
                    
                    if has_person:
                        valid_frames_data.append({
                            'frame': current_batch[idx],
                            'predictions': predictions
                        })
                    else:
                        no_person_counter += 1

                # ====== BATCH RISK DETECTION PART STARTS HERE ====== ====== ====== 
                if valid_frames_data:
                    processed_batches_counter += 1
                    
                    batch_score, rep_image, is_risky, metadata = utils.analyze_ppe_risk_batch(
                        valid_frames_data, threshold = Config.IOU_THRESHOLD
                    )

                    # Store results in our rolling window buffer
                    event_buffer.append({
                        'is_risky': is_risky,
                        'rep_image': rep_image,
                        'score': batch_score,
                        'metadata': metadata
                    })

                    if len(event_buffer) == Config.WINDOW_SIZE:
                        risky_batches_count = sum(1 for b in event_buffer if b['is_risky'])
                        risk_ratio = risky_batches_count / Config.WINDOW_SIZE

                        if risk_ratio >= Config.RISK_THRESHOLD:
                            logger.warning(f"CRITICAL: Confirmed PPE Risk Event! (Ratio: {risk_ratio:.2%})")

                            # Select the 1st, Middle (6th), and 12th representative images
                            # Note: Indexing for 12 items -> 0, 5, 11
                            all_risky_data = [
                                {'rep_image': b['rep_image'], 'metadata': b['metadata']} 
                                for b in event_buffer if b['rep_image'] is not None
                            ]
                            if all_risky_data:
                                selected_data = [
                                    all_risky_data[0],
                                    all_risky_data[len(all_risky_data)//2],
                                    all_risky_data[-1]
                                ]

                                event_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

                                save_thread = threading.Thread(
                                    target = utils.save_event_assets, 
                                    args=(selected_data, event_id)
                                )
                                save_thread.start()
                            else:
                                logger.warning(" - - - - T h e r e  i s  n o t h i n g  t o  i n s p e c t  l a d ")
                                continue

                            # COOLDOWN/RESET: Clear buffer to wait for a completely new set of 12 batches
                            event_buffer = []
                            logger.info("Event sent. Buffer reset for cooldown.")
                        else:
                            # SLIDING WINDOW: If no event triggered, remove the oldest to keep moving
                            event_buffer.pop(0)

                    # ====== BATCH RISK DETECTION PART IS OVER ====== ====== ====== 

            elif not streamer.is_running and len(streamer.frame_buffer) == 0:
                break
            else:
                # Wait briefly if buffer is not ready yet
                time.sleep(0.005)

        # Process remaining batches by the end of the stream
        if event_buffer:
            final_size = len(event_buffer)
            risky_count = sum(1 for b in event_buffer if b['is_risky'])
            final_ratio = risky_count / final_size

            if final_ratio >= Config.RISK_THRESHOLD:
                logger.warning(f"End of Stream Risk Detected! (Ratio: {final_ratio:.2%})")
                
                all_risky_data = [
                    {'rep_image': b['rep_image'], 'metadata': b['metadata']} 
                    for b in event_buffer if b['rep_image'] is not None
                ]

                if all_risky_data:
                    selected_data = [
                        all_risky_data[0],
                        all_risky_data[len(all_risky_data)//2],
                        all_risky_data[-1]
                    ]

                    event_id = f"EOS_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

                    save_thread = threading.Thread(
                        target=utils.save_event_assets, 
                        args=(selected_data, event_id)
                    )
                    save_thread.start()

    except KeyboardInterrupt:
        logger.info("Manual stop triggered.")
    finally:
        streamer.is_running = False
        if hasattr(cap, 'release'):
            cap.release()

    print()
    print("-" * 30)
    logger.info("PIPELINE SUMMARY:")
    logger.info(f"Total Blurry frames skipped: {streamer.blur_counter}")
    logger.info(f"Total Static frames skipped (No Motion): {streamer.motion_skip_counter}")
    logger.info(f"Total Frames skipped (No Person detected): {no_person_counter}")
    logger.info(f"Total Valid batches processed: {processed_batches_counter}")
    logger.info(f"Latency: {time.perf_counter() - start_time}")
    print("-" * 30)


if __name__ == "__main__":
    main()

# TODO: Ondan sonra dockerize edeceğiz. 
# TODO: Volume mounting kısmını ayarladıktan sonra detection işi bitmiş olacak.