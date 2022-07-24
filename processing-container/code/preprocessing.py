
import os
from PIL import Image
import numpy as np
import cv2
import argparse
from io import BytesIO
from tqdm import tqdm

parser = argparse.ArgumentParser(description=
                                 'Reads source images from an s3 bucket, '
                                 'applies preprocessing transformations to them,'
                                 'and saves them back to s3.')

#parser.add_argument('bucket', type=str)
parser.add_argument('--src_prefix', type=str, help='Prefix/folder for source files.')
parser.add_argument('--dest_prefix', type=str, help='Prefix/folder for preprocessed files.')
parser.add_argument('-is', '--img_size', type=int, help='Size to set images to. Defaults to 1800')
parser.add_argument('-bt', '--bin_thresh', type=int, help='Binary threshold for image smoothening. Defaults to 180')

args = parser.parse_args()


def process_img(pil_img, img_size, bin_thresh):
    if img_size is None:
        img_size = 1800
    if bin_thresh is None:
        bin_thresh = 180
    open_cv_image = np.array(pil_img) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    img_resized = set_image_size(open_cv_image, img_size)
    im_new = remove_noise_and_smooth(img_resized, bin_thresh)
    return im_new


def set_image_size(img, img_size):
    img = Image.fromarray(img)
    length_x, width_y = img.size
    factor = max(1, int(img_size / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    img_resized = img.resize(size, Image.ANTIALIAS)
    return img_resized


def image_smoothening(img, bin_thresh):
    ret1, th1 = cv2.threshold(img, bin_thresh, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(img, bin_thresh):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img, bin_thresh)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def main(args):
    input_data_path = os.path.join("/opt/ml/processing/input", args.src_prefix)
    output_data_path = os.path.join("/opt/ml/processing/output", args.dest_prefix)
    
    print(f"Input Data Path {input_data_path}")
    print(f"Output Data Path {output_data_path}")
    
    try:
        os.mkdir(output_data_path)
    except OSError as error:
        print(error) 
    
    for img_obj in os.listdir(input_data_path):
        if img_obj.lower().endswith('.jpg'):
            # process image
            img_fname = f"{input_data_path}/{img_obj}"
            img = Image.open(img_fname)
            processed_img = process_img(img, args.img_size, args.bin_thresh)

            # save preprocessed image back to s3
            img_filename = img_obj.split('/')[-1]
            page_img_obj = f'{output_data_path}/{img_filename}'
            Image.fromarray(processed_img).save(page_img_obj, format='JPEG')


if __name__ == "__main__":
    main(args)

