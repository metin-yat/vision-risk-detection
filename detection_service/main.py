from data_loader import get_source
from config import Config
import logging, cv2, os, utils, time
from dotenv import load_dotenv
from statistics import mean
import threading

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

    try:
        while streamer.is_running or len(streamer.frame_buffer) > 0:
            current_batch = streamer.get_batch()
            
            if current_batch:
                # 1. Ensure the batch is a list of numpy arrays
                # Using image=current_batch is usually enough if the arrays are standard
                try:
                    results = model.infer(image=current_batch,
                            confidence=Config.CONFIDENCE_THRESHOLD)
                except Exception as e:
                    logger.error(f"Inference failed: {e}")
                    continue

                # 2. Filter for human detection
                valid_frames_data = []

                for idx, res in enumerate(results):
                    # Robustly get predictions
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

                # 3. Handle the valid data
                if valid_frames_data:
                    processed_batches_counter += 1
                    # TODO: Send valid_frames_data to Vectorized IoU function
                    # logger.info(f"Batch {processed_batches_counter} contains humans. Processing...")
                    
                # Break after first valid batch for testing if desired
                # break

            elif not streamer.is_running and len(streamer.frame_buffer) == 0:
                break
            else:
                # Wait briefly if buffer is not ready yet
                time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Manual stop triggered.")
    finally:
        streamer.is_running = False
        if hasattr(cap, 'release'):
            cap.release()

    print("-" * 30)
    logger.info("PIPELINE SUMMARY:")
    logger.info(f"Total Blurry frames skipped: {streamer.blur_counter}")
    logger.info(f"Total Static frames skipped (No Motion): {streamer.motion_skip_counter}")
    logger.info(f"Total Frames skipped (No Person detected): {no_person_counter}")
    logger.info(f"Total Valid batches processed: {processed_batches_counter}")
    print("-" * 30)

# TODO: sornasında Vectoriezed IoU risk hesaplatan bir fonksiyona yollasın riskli ise orta kareyi dönsün 
# -> shared_memory queue'ya frame'i ve bilgileri yolla. 
#       

if __name__ == "__main__":
    main()