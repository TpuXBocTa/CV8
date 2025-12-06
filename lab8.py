import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DATA_DIR = "data"

LEFT_IMG_NAME = "scene1.row3.col2.ppm"
CENTER_IMG_NAME = "scene1.row3.col3.ppm"
RIGHT_IMG_NAME = "scene1.row3.col4.ppm"

DISP_CENTER_RIGHT_PATH = "disp_center_right.png"
DISP_CENTER_LEFT_PATH = "disp_center_left.png"
DISP_FUSED_PATH = "disp_center_fused.png"
DEPTH_FUSED_PATH = "depth_center_fused.png"
OVERLAY_FUSED_PATH = "overlay_center_fused.png"


def compute_disparity(left_gray, right_gray, num_disparities=192, block_size=3):
    if num_disparities % 16 != 0:
        raise ValueError("num_disparities must be multiple of 16")

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disp_raw

def disparity_to_color(disp_raw):
    valid = disp_raw > 0
    vals = disp_raw[valid]
    if vals.size == 0:
        return None, valid

    d_min = np.percentile(vals, 5)
    d_max = np.percentile(vals, 95)
    if d_max == d_min:
        d_max = d_min + 1e-6

    disp_clipped = np.clip(disp_raw, d_min, d_max)
    disp_norm = np.zeros_like(disp_clipped, dtype=np.float32)
    disp_norm[valid] = (disp_clipped[valid] - d_min) / (d_max - d_min)

    disp_img = (disp_norm * 255).astype(np.uint8)
    disp_img[~valid] = 0

    disp_color = cv2.applyColorMap(disp_img, cv2.COLORMAP_INFERNO)
    return disp_color, valid

def show_3d_filtered_surface(depth_rel, rgb_image, step=3, depth_threshold=0.02):
    valid = depth_rel > 0

    Z = depth_rel[::step, ::step]
    V = valid[::step, ::step]
    RGB = rgb_image[::step, ::step, ::-1].astype(np.float32) / 255.0

    Y, X = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]

    faces = []
    colors = []

    for y in range(Z.shape[0] - 1):
        for x in range(Z.shape[1] - 1):

            if not (V[y, x] and V[y+1, x] and V[y, x+1] and V[y+1, x+1]):
                continue

            z_vals = [
                Z[y, x], Z[y+1, x], Z[y+1, x+1], Z[y, x+1]
            ]

            if max(z_vals) - min(z_vals) > depth_threshold:
                continue

            quad = [
                [X[y, x],     Y[y, x],     Z[y, x]],
                [X[y+1, x],   Y[y+1, x],   Z[y+1, x]],
                [X[y+1, x+1], Y[y+1, x+1], Z[y+1, x+1]],
                [X[y, x+1],   Y[y, x+1],   Z[y, x+1]],
            ]

            face_color = np.mean([
                RGB[y, x], RGB[y+1, x], RGB[y+1, x+1], RGB[y, x+1]
            ], axis=0)

            faces.append(quad)
            colors.append(face_color)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(faces, facecolors=colors, linewidths=0.2, edgecolors='k')
    ax.add_collection3d(poly)

    ax.set_xlim(0, X.shape[1])
    ax.set_ylim(0, Y.shape[0])
    ax.set_zlim(np.min(Z[V]), np.max(Z[V]))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

def main():
    left_path = os.path.join(DATA_DIR, LEFT_IMG_NAME)
    center_path = os.path.join(DATA_DIR, CENTER_IMG_NAME)
    right_path = os.path.join(DATA_DIR, RIGHT_IMG_NAME)

    left = cv2.imread(left_path)
    center = cv2.imread(center_path)
    right = cv2.imread(right_path)

    h = min(left.shape[0], center.shape[0], right.shape[0])
    w = min(left.shape[1], center.shape[1], right.shape[1])
    left = left[:h, :w]
    center = center[:h, :w]
    right = right[:h, :w]

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
    center_gray = cv2.GaussianBlur(center_gray, (5, 5), 0)
    right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

    disp_CR = compute_disparity(center_gray, right_gray)
    disp_CR_color, _ = disparity_to_color(disp_CR)
    cv2.imwrite(DISP_CENTER_RIGHT_PATH, disp_CR_color)

    center_flip = cv2.flip(center_gray, 1)
    left_flip = cv2.flip(left_gray, 1)
    disp_CF_LF = compute_disparity(center_flip, left_flip)
    disp_CL = cv2.flip(disp_CF_LF, 1)
    disp_CL_color, _ = disparity_to_color(disp_CL)
    cv2.imwrite(DISP_CENTER_LEFT_PATH, disp_CL_color)

    disp1 = disp_CR.copy()
    disp2 = disp_CL.copy()
    disp1[disp1 <= 0] = np.nan
    disp2[disp2 <= 0] = np.nan

    fused = np.nanmedian(np.stack([disp1, disp2], axis=0), axis=0)
    fused_valid = ~np.isnan(fused)
    fused[~fused_valid] = 0

    fused_color, fused_valid_mask = disparity_to_color(fused)
    cv2.imwrite(DISP_FUSED_PATH, fused_color)

    eps = 1e-6
    depth_rel = np.zeros_like(fused, dtype=np.float32)
    depth_rel[fused_valid_mask] = 1.0 / (fused[fused_valid_mask] + eps)

    depth_vals_all = depth_rel[fused_valid_mask]

    z_far = np.percentile(depth_vals_all, 95)
    far_mask = (depth_rel > z_far) & fused_valid_mask
    depth_rel[far_mask] = 0
    fused_valid_mask[far_mask] = False

    depth_rel_to_show_3d = depth_rel

    show_3d_filtered_surface(depth_rel_to_show_3d, center, step=3, depth_threshold=0.0075)

    fused_disp_for_overlay = fused_color if fused_color is not None else np.zeros_like(center)
    overlay = cv2.addWeighted(center, 0.6, fused_disp_for_overlay, 0.4, 0)
    cv2.imwrite(OVERLAY_FUSED_PATH, overlay)

    cv2.imshow("Center", center)
    cv2.imshow("Disp fused", fused_disp_for_overlay)
    cv2.imshow("Overlay fused", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
