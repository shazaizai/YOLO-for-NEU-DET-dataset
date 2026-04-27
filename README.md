# YOLO-for-NEU-DET-dataset FastAPI：
This is a code repository for object detection on the NEU-DET dataset using YOLOv26. The project also provides a template for calling the YOLOv26 ONNX model using FastAPI.

[中文版 README](./README_zh.md)
[English README](./README.md)

# Example:
The following is a sample of some content from this project:

<img width=600 src="./FastAPI_sample.png" alt="">

---
# Start

## Local deployment:
Deploying this project locally requires the following steps:

1. Install the software package:
```
pip install -r requirements.txt

``` 
2. Start the application:

```
python app.py
```

# FastAPI documentation address:
http://localhost:8080/docs

<img width=600 src="./FastAPI_doc.png" alt="">

---
# 🚀Code example:

### example1：
The following code demonstrates how to use the ONNX model for object detection inference and returns the detection results:
```python
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
        print(e)
        return None
```
output：
```
[[x,y,x,y],conf,cls]
```
In addition, the project also has a method for inference using YOLO's built-in tools, ModelInferenceByYOLO(), with the same input parameters as ModelInference().
Of course, the project also provides methods for custom invocation:
```python
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
        
        return None
```
output：
```
[[x,y,x,y],conf,cls]
```
Function calls can be made using Inference_method=YOLO or by using the default method.

### Example 2: Plotting the detection results on the image
The following code demonstrates how to plot the detection results on an image.
```python
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
```
# Future Prospects

1. Deploy ONNX models on different platforms.

2. Accelerate TensorRT model inference.

3. Video detection functionality.

4. Model quantization.

5. .........