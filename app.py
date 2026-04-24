from fastapi import FastAPI,File,UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
from toolapi import get_model_pridict
from toolapi import add_bboxs_on_img
from PIL import Image
from io import BytesIO

app = FastAPI()


@app.get("/")
def hello():
    return {"Hello": "World"}


@app.post("/imgobject")
def image_pridict(file: bytes = File(...)):
    # contents = file.file.read()
    contents=Image.open(BytesIO(file)).resize((224,224))
    label=['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    model_path=r"ultralytics\best.onnx"
    result=get_model_pridict(model_path,"cpu",contents)

    output_image=add_bboxs_on_img(label,contents,result)

    # 将 PIL Image 转为 bytes
    img_byte_arr = BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(BytesIO(img_byte_arr), media_type="image/png")
    
    # return output_image




if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

