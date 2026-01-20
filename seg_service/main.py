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
    logger.info("Segmentation Service starting...")

    # loading the model from Roboflow. 
    model = utils.initialize_segmentation_model(Config.MODEL_PATH,
                                            api_key)
    
    if not utils.test_segmentation_model(model=model):
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
    try:
        while streamer.is_running or len(streamer.frame_buffer) > 0:
            current_batch = streamer.get_batch()
            
            if current_batch:
                print(f"--- Processing batch of {len(current_batch)} clean frames ---")
                results = model.infer(current_batch)
                print(f"Batch inference complete. Received {len(results)} results.")
                if results:
                    first_frame = results[0]
                    # Roboflow structure: predictions is usually a list of objects found
                    predictions = getattr(first_frame, 'predictions', [])
                    
                    print("-" * 40)
                    print(f"FIRST FRAME SUMMARY ({len(predictions)} objects found):")
                    
                    if not predictions:
                        print("No objects detected in this frame.")
                    else:
                        for i, pred in enumerate(predictions):
                            # Extracting class and confidence
                            cls = getattr(pred, 'class_name', 'Unknown')
                            conf = getattr(pred, 'confidence', 0.0)
                            print(f" {i+1}. [{cls}] - Confidence: {conf:.2%}")
                    print("-" * 40)

                # Break after first batch as you requested
                break

            # If stream ended but we have a leftover frames, less than BATCH_LENGTH frames
            elif not streamer.is_running and len(streamer.frame_buffer) > 0:
                # print(f"Stream ended. Processing leftover frames: {len(streamer.frame_buffer)}")
                leftover_batch = list(streamer.frame_buffer)
                streamer.frame_buffer.clear() # Clear it to break the while loop
                results = model.infer(current_batch)
                print(f"Batch inference complete. Received {len(results)} results.")

            else:
                # Wait briefly if buffer is not ready yet
                time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Manual stop triggered.")
    finally:
        streamer.is_running = False
        if hasattr(cap, 'release'):
            cap.release()

    logger.info(f"Image Pipeline completed. There was {streamer.blur_counter} blurry frames get skipped.")


# TODO: sornasında Vectoriezed IoU risk hesaplatan bir fonksiyona yollasın riskli ise orta kareyi dönsün 
# -> shared_memory queue'ya frame'i ve bilgileri yolla. 
#       

if __name__ == "__main__":
    main()