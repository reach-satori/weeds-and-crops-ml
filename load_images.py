import cv2
import pandas as pd
from math import atan2,degrees
import numpy as np
from os.path import join
from multiprocessing import Pool, cpu_count
from matplotlib import pyplot as plt


DROPBOX_LOCATION = "./dropbox_things/DroneImages/"
CSV_LOCATION = "./square_coords.csv"
CH_NAMES = ["red", "green", "blue", "red edge", "nir"]
IMAGE_MARGIN = 50  # margin around each calibration point

def _normalize_img(im):
    mean, std = np.nanmean(im), np.nanstd(im)
    return (im - mean)/std

def _get_widthnangle(df):
    angles = []
    widths = []
    heights = []
    for i, row in df.iterrows():
        atop = atan2(row["tr_y"] - row["tl_y"], row["tr_x"] - row["tl_x"])
        abot = atan2(row["br_y"] - row["bl_y"], row["br_x"] - row["bl_x"])
        aleft = atan2(row["tl_y"] - row["bl_y"], row["tl_x"] - row["bl_x"])
        aright = atan2(row["tr_y"] - row["br_y"], row["tr_x"] - row["br_x"])
        angles.append(np.mean([atop, abot, aright + np.pi/2, aleft + np.pi/2]))
        widths.append(np.linalg.norm([row["tr_x"] - row["tl_x"],row["tr_y"] - row["tl_y"]]))
        heights.append(np.linalg.norm([row["br_x"] - row["tr_x"],row["br_y"] - row["tr_y"]]))
    df["angle"] = pd.Series(angles)  # angle from the top left calibration square to the top right
    df["width"] = pd.Series(widths)  # sidelen from top left square to the top right
    df["height"] = pd.Series(heights)  # sidelen from top right square to the bottom right
    return df

def _transform_dfpoints(row, mat):
    points = iter(row.drop(["fileprefix", "width","angle", "height"]).index)
    for x in points:
        y = next(points)
        arr = np.array([[row[x]],[row[y]],[1.]], dtype=np.float32)
        nx, ny = np.matmul(mat, arr)
        row[x], row[y] = nx, ny
    return row

def _preprocess_single_image(row, minwidth, minheight, image_location):
    channels = []
    paths = [join(image_location, f"{row['fileprefix']}_{channel}.tif") for channel in CH_NAMES]
    for fp in paths:
        channels.append(cv2.imread(fp, cv2.IMREAD_ANYDEPTH))

    img = cv2.merge(channels)
    img[img < 0] = float("nan")  # -10000 empty pixels become NaN
    img = _normalize_img(img)

    # rotate around the top left calibration point
    ctr = tuple(i.item() for i in row[["tl_x", "tl_y"]].values)
    rotmat = cv2.getRotationMatrix2D(center=ctr,
                                     angle=degrees(row["angle"]),
                                     scale=minwidth/row["width"])

    row = _transform_dfpoints(row, rotmat)  # record rotation in the dataframe

    rotated = cv2.warpAffine(img,  # rotate image itself
                             M=rotmat,
                             dsize=(int(row["tr_x"])+2*IMAGE_MARGIN,
                                    int(row["bl_y"]+2*IMAGE_MARGIN)))
    transmat = np.array([[1, 0, -int(row["tl_x"] - IMAGE_MARGIN)],  # translation matrix
                         [0, 1, -int(row["tl_y"] - IMAGE_MARGIN)]], dtype=np.float32)

    row = _transform_dfpoints(row, transmat)  # apply in dataframe

    rotated = cv2.warpAffine(rotated,  # apply translation to image
                             M=transmat,
                             dsize=(int(row["br_x"]+IMAGE_MARGIN), int(row["br_y"]+IMAGE_MARGIN)))

    print(rotated.shape)
    # final size adjustment, resizes only by ~20 pixels max.
    rotated = cv2.resize(rotated,
                         (int(minwidth + 2*IMAGE_MARGIN), int(minheight + 2*IMAGE_MARGIN)),
                         interpolation=cv2.INTER_CUBIC)
    print(rotated.shape)

    return rotated

def load_images(image_location=DROPBOX_LOCATION, csv_location=CSV_LOCATION, multiprocess=False):
    """
    image_location: location of the drone image folder.
    csv_location: location of square_coords.csv
    multiprocess: runs the preprocessing parallel if true. Takes up a lot of memory(~10gb)
    since it loads every image at the same time. If false, yields images one by one instead.

    returns: list of numpy arrays (width, height, 5). Note that the widths and heights are not
    all the exact same.
    Images are scaled to the smallest image, rotated so the two top squares of the crop are level,
    and normalized with mean 0 and std deviation 1.
    """
    calib = pd.read_csv(csv_location)
    calib = _get_widthnangle(calib)

    smallrow = calib["width"].idxmin()
    minwidth = calib.iloc[smallrow]["width"]
    minheight = calib.iloc[smallrow]["height"]  # For scaling each image to the smallest image.
    calib.drop(index=6, axis=0, inplace=True)  # There's an image where the crop gets cut off.
    if multiprocess:
        with Pool(cpu_count()) as p:
            out = p.starmap(_preprocess_single_image,
                            [(calib.iloc[i], minwidth, minheight, image_location) for i in range(len(calib))])
            return out
    else:
        for i in range(len(calib)):
            row = calib.iloc[i]
            yield _preprocess_single_image(row, minwidth, minheight, image_location)
        return

def block_breakup(img):
    lines = open("plot_corners.csv", "r").readlines()
    coords = []
    h, w, _ = img.shape
    for line in lines:
        line = tuple(int(i) for i in (line.rstrip().split(",")))
        coords.append(line)
    coords = np.array(coords, dtype=np.int32)
    leftside, rightside = coords[0::2], coords[1::2]  # csv file alternates left and right plots
    xleft, xright = int(np.mean(leftside[:,0])), int(np.mean(rightside[:,0]))

    prevy = 0
    for pt in reversed(leftside):
        yield img[prevy:pt[1], 0:xleft, :]
        prevy = pt[1]
    prevy = 0
    for pt in reversed(rightside):
        yield img[prevy:pt[1], xright:, :]
        prevy = pt[1]


if __name__ == "__main__":
    imgs = load_images()
    for img in imgs:
        img = img[...,:3].copy()
        img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        for im in block_breakup(img):
            plt.imshow(im)
            plt.show()
        # plt.imshow(img[...,:3])
        # plt.show()
