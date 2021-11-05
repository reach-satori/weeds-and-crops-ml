import cv2
import pandas as pd
from math import atan2,degrees
import numpy as np
from multiprocessing import Pool, cpu_count

DROPBOX_LOCATION = "./dropbox_things/DroneImages/"
CH_NAMES = ["red", "green", "blue", "red edge", "nir"]
IMAGE_MARGIN = 300  # margin around each calibration point

def _normalize_img(im):
    mean, std = np.nanmean(im), np.nanstd(im)
    return (im - mean)/std

def _get_widthnangle(df):
    angles = []
    widths = []
    for i, row in df.iterrows():
        angles.append(atan2(row["tr_y"] - row["tl_y"],
                            row["tr_x"] - row["tl_x"]))
        widths.append(np.linalg.norm([row["tr_x"] - row["tl_x"],row["tr_y"] - row["tl_y"]]))
    df["angle"] = pd.Series(angles)  # angle from the top left calibration square to the top right
    df["width"] = pd.Series(widths)  # width from top left calibration square to the top right
    return df

def _transform_dfpoints(row, mat):
    points = iter(row.drop(["fileprefix", "width","angle"]).index)
    for x in points:
        y = next(points)
        arr = np.array([[row[x]],[row[y]],[1.]], dtype=np.float32)
        nx, ny = np.matmul(mat, arr)
        row[x], row[y] = nx, ny
    return row

def _preprocess_single_image(row, minwidth):
    channels = []
    for fp in [f"{DROPBOX_LOCATION}{row['fileprefix']}_{channel}.tif" for channel in CH_NAMES]:
        channels.append(cv2.imread(fp, cv2.IMREAD_ANYDEPTH))

    img = cv2.merge(channels)
    img[img < 0] = float("nan")  # -10000 empty pixels become NaN
    img = _normalize_img(img)

    # rotate around the top left calibration point
    rotmat = cv2.getRotationMatrix2D(center = tuple(row[["tl_x", "tl_y"]].values),
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
    return rotated

def load_images():
    calib = pd.read_csv("square_coords.csv")
    calib = _get_widthnangle(calib)

    minwidth = calib["width"].min()  # For scaling each image to the smallest image.
    calib.drop(index=6, axis=0, inplace=True)  # There's an image where the crop gets cut off.
    with Pool(cpu_count()) as p:
        out = p.starmap(_preprocess_single_image,
                        [(calib.iloc[i], minwidth) for i in range(len(calib))])
    return out


if __name__ == "__main__":
    load_images()
