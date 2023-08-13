from huffman import HuffmanCoding
from arithmetic import Arithmetic
from utils import *
import time
import argparse
parser = argparse.ArgumentParser(description="Image processing script")
parser.add_argument("color_space", choices=["rgb", "hsv", "ycbcr"], help="Color space to use for processing")
parser.add_argument("coding", choices=["huff", "arth"], help="coding")
parser.add_argument("image_path", help="Name of the input image file")
parser.add_argument("output_path", help="Path to save the processed image")
args = parser.parse_args()

Coding = {
    "huff": HuffmanCoding,
    "arth": Arithmetic,
}

tic = time.time()
img, img_size = read_image(args.image_path)
changed_color = change_color_map(img, f"rgb2{args.color_space}")
freq_img = dct_img(changed_color)
quantized_img = quantize(freq_img, args.color_space)
flattened_img, shape = flatten(quantized_img)
coding = Coding[args.coding](flattened_img)
encoded_img = coding.encode_arr(flattened_img)
print(f"encode:{len(encoded_img)} bits vs origina:{img_size} bits")
decoded_img = coding.decode_arr(encoded_img, len(flattened_img))
reshaped_img = unflatten(decoded_img, shape)
dequantized_img = dequantize(reshaped_img, args.color_space)
spatial_img = idct_freq_img(dequantized_img)
reconstructed_img = change_color_map(spatial_img, f"{args.color_space}2rgb")
print("psnr: ",calculate_psnr(reconstructed_img, img), "dB")
save_image(args.output_path + f"{args.color_space}_{args.coding}.png", reconstructed_img)
print("time: ", time.time() - tic)