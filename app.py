from yolodetect import ONNX_Inference as OI
from PIL import Image
import io
import pandas as pd
import numpy as np
from ultralytics.utils.plotting import Annotator, colors



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
    try:
        if Inference_method=="YOLO":
            return OI.ModelInferenceByYOLO(Onnxmodel_path,providers,input_image,confidience,imgs)
        else:
            return OI.ModelInference(Onnxmodel_path,providers,input_image,confidience,imgs)
    except Exception as e:
        print(e)
        # print("get_model_pridict")
        return None
    
def add_bboxs_on_img(label,images,Inference_result):
    """
    """
    # Create an annotator object
    try:

        annotator=Annotator(np.array(images))

        #Inference_result [[x, y, w, h], conf, class]
        for r in Inference_result:
            bbox=r[0]
            text= f"{label[int(r[2])]}: {int(r[1]*100)}%"
            annotator.box_label(bbox,text,color=colors(r[2],True))
        
        return Image.fromarray(annotator.result()) 
    except Exception as e:
        # print(e)
        print("add_bboxs_on_img",e)
        return None




