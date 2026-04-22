from yolodetect import ONNX_Inference as OI
from PIL import Image
import io
import pandas as pd
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


names=['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

#bytes2image
def bytes2image(binary_image):
    """
    :param : binary_image is input image
    :return : Image for YOLO

    """
    input_image=Image.open(io.BytesIO(binary_image)).convert('RGB')
    return input_image


def get_model_pridict(Onnxmodel_path,providers,input_image,confidience=0.5,imgs=224,Inference_method='Normal'):
    """
    """
    if Inference_method=="YOLO":
        return OI.ModelInferenceByYOLO(Onnxmodel_path,providers,input_image,confidience,imgs)
    else:
        return OI.ModelInference(Onnxmodel_path,providers,input_image,confidience,imgs)
    
def add_bboxs_on_img(label,images,Inference_result):
    """
    """
    # Create an annotator object
    annotator=Annotator(np.array(images))

    #Inference_result [[x, y, w, h], conf, class]
    for r in Inference_result:
        bbox=r[0]
        text= f"{label[int(r[2])]}: {int(r[1]*100)}%"
        annotator.box_label(bbox,text)
    
    return Image.fromarray(annotator.result())




