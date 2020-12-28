from PIL import Image
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

def imageFromURL(url):
    response = requests.get(url)
    image_bytes = BytesIO(response.content)

    img = Image.open(image_bytes)
    return np.array(img)

