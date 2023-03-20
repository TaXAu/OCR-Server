import io
import base64
from PIL import Image
import numpy as np
from paddleocr import PPStructure
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
from typing import Tuple

OCRApp = FastAPI()
StructureEngine = PPStructure(show_log=True, image_orientation=True)


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
    html: str
    bbox: Tuple[int, int, int, int]


@OCRApp.post("/api", response_model=ResponseModel)
async def api(body: RequestBodyModel):
    image_bytes = base64.b64decode(body.base64str)
    image_arr = np.array(Image.open(io.BytesIO(image_bytes)))
    # if arr is 4 channel, convert to 3 channel
    if image_arr.shape[2] == 4:
        image_arr = image_arr[:, :, :3]

    result: list[dict] = StructureEngine(image_arr)

    # get fist table result
    result0 = result[0]
    # get html
    raw_html = result0["res"]["html"]
    frame = pd.read_html(raw_html)
    html = frame[0].to_html()

    return {
        "html": html,
        "bbox": result0["bbox"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(OCRApp, host="localhost", port=8000)
