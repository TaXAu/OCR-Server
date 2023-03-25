import io
import base64
from PIL import Image
import numpy as np
from paddleocr import PPStructure, save_structure_res
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
from typing import Tuple, Dict, Any
import logging
import time

OCRApp = FastAPI()
StructureEngine = PPStructure(show_log=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='ocr.log',
                    filemode='a')



@OCRApp.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><lang="en">
    <head><style>
    h1 {
    margin: 4em;
    text-align: center;
    }
    </style></head>
    <body>
        <h1>TaXAu OCR is Running</h1>
    </body>
    </html>
    """


class RequestBodyModel(BaseModel):
    base64str: str


class ResponseModel(BaseModel):
    data: Dict
    bbox: Tuple[int, int, int, int]


@OCRApp.post("/api", response_model=ResponseModel)
async def api(body: RequestBodyModel):
    logging.info("Received request")
    logging.info(body.base64str[:100])
    image_bytes = base64.b64decode(body.base64str)
    image_arr = np.array(Image.open(io.BytesIO(image_bytes)))
    # if arr is 4 channel, convert to 3 channel
    if image_arr.shape[2] == 4:
        image_arr = image_arr[:, :, :3]

    result: list[dict] = StructureEngine(image_arr)

    id = time.time_ns()
    save_structure_res(result, f"./output/{id}", f"{id}")

    # get fist table result
    result0 = result[0]
    # get html
    raw_html = result0["res"]["html"]
    frame = pd.read_html(raw_html)
    frame[0].fillna("", inplace=True)
    data = frame[0].to_dict()
    return {
        "data": data,
        "bbox": result0["bbox"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(OCRApp, host="localhost", port=8001)
