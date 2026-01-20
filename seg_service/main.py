from data_loader import get_source
import config
from ultralytics import YOLO

# Get video source
cap = get_source(config.RAW_IMAGES_DIR)
# Load the model

# TODO: Modeli yükleyen ve dummy bir siyah foto ile warm up eden fonksiyonu yaz
#       model = FUNCTION(.)
model = YOLO(config.MODEL_PATH)

# TODO: Filter the Images/Frames respect to their how much blur they posses (Laplacian 
#       Filtering)->  Scale the images properly to fixed sizes ->  
#       Batch kadar frame bekle, eğer bir süre gelmezse, deadlock ihtiamlini önle  [OPTIONAL] -> 
#       Batch haline getir (10 frame/image =~ 0.3 second) -> 
#       segmentation fonksiyonu yaz -> sornasında Vectoriezed IoU risk hesaplatan 
#                                   bir fonksiyona yollasın ->
#       riskli ise orta kareyi al -> shared_memory queue 
#       
# TODO: LOGGING ile log ver: DEBUG: Görüntü boyutu O X P boyutlandırıldı. (geliştirme aşamaası)
#                            INFO: Kamera Bağlantısı kuruldu (genel akış)
#                            WARNING: Kare atlaması yaşandı ama devam ediyorum. (dikkat et.)
#                            ERROR: Kamera bağlantısı koptu (işlem durdu.)
#

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TODO: YOLOv11 inference