import onnxruntime as ort
import numpy as np
from PIL import Image
from ultralytics import YOLO

#Load model 
def LoadOnnxmodel(Onnxmodel_Path,Providers):
    """
    :param: Onnxmodel_Path: model path.模型位置
    :param: Providers: （[‘CUDAExecutionProvider’,’CPUExecutionProvider’]）
    """
    provider_list = ['CUDAExecutionProvider'] if Providers == 'cuda' else ['CPUExecutionProvider']
  
    return ort.InferenceSession(Onnxmodel_Path, providers=provider_list)

#图像处理
def ImageProcess(Image_path,imgs):
    """
    :param: Image_path:image path
    :param: imgs: image size like 640*640
    """
    img = Image.open(Image_path).resize((imgs, imgs))
    return np.array(img).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0


# 推理
def ModelInference(Onnxmodel_path,providers,Image_path,confidence_threshold=0.5,imgs=224):
    """
    :param: Onnxmodel_Path: model path.模型位置
    :param: Providers: cuda cpu
    :param: Image_path:image path
    :param: confidence_threshold:defult 0.5
    :param: imgs: image size like 640*640
    :return: [[x,y,x,y],conf,cls]
    """
    try:
        session=LoadOnnxmodel(Onnxmodel_path,providers)
        img_array=ImageProcess(Image_path,imgs)
        outputs=session.run(None, {"images": img_array})
        predictions = outputs[0][0]  # 输出形状: (300, 6)
        confidence_threshold = confidence_threshold
        # predictions [x, y, w, h, conf, class] choose conf>confidence_threshold
        results=predictions[predictions[:, 4] > confidence_threshold]

        boxb=[]
        for r in results:
            boxarray=[]
            boxxyxy=r[0:4].tolist()
            boxconf=r[4].tolist()
            boxcls=r[5].tolist()
            boxarray.append(boxxyxy)
            boxarray.append(boxconf)
            boxarray.append(boxcls)
            boxb.append(boxarray)

        return boxb
    except Exception as e:
        return None

#yolo inference
def ModelInferenceByYOLO(Onnxmodel_path,providers,Image_path,confidence_threshold=0.5,imgs=224):
    """
    :param: Onnxmodel_Path: model path.模型位置
    :param: Providers: cuda cpu
    :param: Image_path:image path
    :param: confidence_threshold:defult 0.5
    :param: imgs: image size like 640*640
    :return: [[x,y,x,y],conf,cls]
    """
    try:
        model = YOLO(Onnxmodel_path, task='detect')
        results=model.predict(
            source=Image_path,
            conf=confidence_threshold,
            iou=0.45,                   # 设置NMS的IoU阈值
            device=providers               # 可选择使用GPU
        )
        array=[]

        for r in results:
            boxes=r.boxes
            for box in boxes:
                arraybox=[]
                arraybox.append(box.xyxy.numpy().tolist()[0])
                arraybox.append(box.conf.numpy().tolist()[0])
                arraybox.append(box.cls.numpy().tolist()[0])
                array.append(arraybox)

        return array
    except Exception as e:
        return None
        