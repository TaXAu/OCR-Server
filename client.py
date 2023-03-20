import base64
import requests as req


# A Post Request Function to call /html API
def post_request(base64str: str):
    req_body = {
        "base64str": base64str
    }
    response = req.post('http://localhost:8000/api', json=req_body)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error"


if __name__ == "__main__":
    image_base64 = base64.b64encode(open('imgs/table/table.jpg', 'rb').read()).decode('utf8')
    result = post_request(image_base64)
    print(result)
