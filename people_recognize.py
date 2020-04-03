import warnings
warnings.filterwarnings("ignore")
from keras.backend import clear_session
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import model.keras_layers
from cv2 import imread,resize

from model.models.keras_ssd512 import ssd_512
from model.keras_loss_function.keras_ssd_loss import SSDLoss

import numpy as np
from matplotlib import pyplot as plt

# Set the image size.
img_height = 512
img_width = 512

weights_path = r'model\weights\VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'

class PersonRecognizer:
    def __init__(self,weights_path=weights_path):
        self.classes = ['background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        
        self.prediction_order = ['class','confidence','xmin','ymin','xmax','ymax']
        
        model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                   two_boxes_for_ar1=True,
                   steps=[8, 16, 32, 64, 128, 256, 512],
                   offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   clip_boxes=False,
                   variances=[0.1, 0.1, 0.2, 0.2],
                   normalize_coords=True,
                   subtract_mean=[123, 117, 104],
                   swap_channels=[2, 1, 0],
                   confidence_thresh=0.5,
                   iou_threshold=0.45,
                   top_k=200,
                   nms_max_output_size=400)
        
        model.load_weights(weights_path, by_name=True)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        
        self.model = model
        
    def detect_people(self,frame,confidence=0.5):
        input_images=[]
        frame = resize(frame,(512,512))
        input_images.append(frame)
        input_images=np.array(input_images)
        y_pred = self.model.predict(input_images)
        confidence_threshold = confidence
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        people_boxes = [box for box in y_pred_thresh[0] if self.classes[int(box[0])]=='person']
        return people_boxes