import numpy as np
import cv2 as cv
from PIL import Image
from scipy.fftpack import dct, idct

LUMA_QUANTIZATION_MATRIX = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)
CHROMA_QUANTIZATION_MATRIX = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)

LUMA_QUANTIZATION_MATRIX2 = np.array(
    [
        [4, 2,  2,  4,  6, 8, 9, 10],
        [3, 3,  3,  4,  6, 9, 12, 9],
        [3, 3,  4,  6, 8, 9, 12, 10],
        [3, 4,  5,  7, 8, 12, 11, 9],
        [4, 5,  8, 9, 11, 15, 14, 12],
        [6, 8, 9, 11, 12, 15, 16, 14],
        [9, 10, 12, 14, 15, 16, 16, 15],
        [10, 12, 13, 14, 16, 14, 14, 14]
        ]
)

CHROMA_QUANTIZATION_MATRIX2 = np.array(
    [
        [ 4,  4,  6, 8, 9, 10, 11, 12],
        [ 4,  5,  6, 9, 11, 11, 12, 12],
        [ 6,  6, 10, 11, 12, 12, 12, 12],
        [8, 9, 10, 11, 12, 12, 12, 12],
        [10, 11, 11, 12, 12, 12, 12, 12],
        [11, 11, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12]
        ]
)

LUMA_MATRIX = {
    "rgb": LUMA_QUANTIZATION_MATRIX,
    "ycbcr": LUMA_QUANTIZATION_MATRIX2,
    "hsv": LUMA_QUANTIZATION_MATRIX2
}

CHROMA_MATRIX = {
    "rgb": CHROMA_QUANTIZATION_MATRIX,
    "ycbcr": CHROMA_QUANTIZATION_MATRIX2,
    "hsv": CHROMA_QUANTIZATION_MATRIX2
}

def read_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    img_size = img.shape[0] * img.shape[1] * 8 * img.shape[2]
    img = img[:8*(img.shape[0] // 8), :8*(img.shape[1] // 8)]
    return img, img_size

def save_image(image_path: str, img: np.ndarray):
    img = Image.fromarray(img.astype(np.uint8))
    img.save(image_path)

def rgb2ycbcr(rgb: np.ndarray):
    matrix = np.array([
        [.299, .587, .114],
        [-.168736, -.331264, .5],
        [.5, -.418688, -.081312]
    ])
    ycbcr = rgb.dot(matrix.T)
    ycbcr[:,:,[1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(ycbcr :np.ndarray):
    matrix = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -.71414],
        [1, 1.772, 0]
    ])
    rgb = ycbcr.astype(np.float64)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(matrix.T)
    rgb = np.maximum(rgb, 0)
    rgb = np.minimum(rgb, 255)
    return np.uint8(rgb)

def rgb2hsv(rgb: np.ndarray):
    hsv =  cv.cvtColor(rgb, cv.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] *= 255 / 179
    return hsv.astype(np.uint8)

def hsv2rgb(hsv: np.ndarray):
    hsv = hsv.astype(np.float32)
    hsv[..., 0] *= 179 / 255
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2RGB)

COLOR2COLOR = {
    "ycbcr2rgb": ycbcr2rgb,
    "rgb2ycbcr": rgb2ycbcr,
    "hsv2rgb": hsv2rgb,
    "rgb2hsv": rgb2hsv,
    "rgb2rgb": lambda x:x,
}

def change_color_map(mat: np.ndarray, mod :str):
    return COLOR2COLOR[mod](mat)

def dct2(a):
    return dct(dct(a.T, norm = 'ortho').T, norm = 'ortho')

def dct_img(img):
    result = np.zeros(img.shape, dtype=np.float64)
    for i in range(img.shape[0] // 8):
        for j in range(img.shape[1] // 8):
            block = np.array(img[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8])
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8] = dct2(block)
    return result

def quantize(img, color_space):
    result = np.zeros(img.shape, dtype=np.int16)
    for i in range(img.shape[0] // 8):
        for j in range(img.shape[1] // 8):
            block = img[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8]
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 0] = np.floor(block[..., 0] / LUMA_MATRIX[color_space])
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 1] = np.floor(block[..., 1] / CHROMA_MATRIX[color_space])
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 2] = np.floor(block[..., 2] / CHROMA_MATRIX[color_space])
    return result

def flatten(img):
    return img.flatten(), img.shape

def unflatten(flattened_img, shape):
    res = np.array(flattened_img)
    return res.reshape(shape)

def dequantize(image, color_space):
    
    result = np.zeros(image.shape, dtype=np.int16)
    for i in range(image.shape[0] // 8):
        for j in range(image.shape[1] // 8):
            block = image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8]
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 0] = block[..., 0] * LUMA_MATRIX[color_space]
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 1] = block[..., 1] * CHROMA_MATRIX[color_space]
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8, 2] = block[..., 2] * CHROMA_MATRIX[color_space]
    return result.astype(np.int16)

def idct2(a):
    return idct(idct(a.T, norm = 'ortho').T, norm = 'ortho')

def idct_freq_img(img):
        
    result = np.zeros(img.shape, dtype=np.float64)
    for i in range(img.shape[0] // 8):
        for j in range(img.shape[1] // 8):
            block = img[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8]
            temp = np.array(block)
            result[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8] = idct2(temp)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = np.iinfo(image1.dtype).max
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr