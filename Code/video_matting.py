import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import gaussian_kde
from tqdm import tqdm


""" matting Variables """
r_matting = 1.2
rho_matting = 10
object_half_size_in_y = 600
object_half_size_in_x = 200


def draw_percent_image_Matting_1(frame, n_frame):
    image = np.ones((100, 600)) / 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'frame ' + str(frame) + ' of ' + str(n_frame) + ' -- ' + str(int((frame / n_frame) * 100)) + '% Done'
    cv2.putText(image, text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow('Matting percent', image)
    cv2.waitKey(2)


def getMissingMask_1(slab):
    nan_mask = np.where(np.isnan(slab), 1, 0)
    if not hasattr(slab, 'mask'):
        mask_mask = np.zeros(slab.shape)
    else:
        if slab.mask.size == 1 and slab.mask == False:
            mask_mask = np.zeros(slab.shape)
        else:
            mask_mask = np.where(slab.mask, 1, 0)
    mask = np.where(mask_mask + nan_mask > 0, 1, 0)

    return mask


def geodesic_1(img, seed):
    seedy, seedx = seed
    mask = getMissingMask_1(img)

    # ----Call distance_transform_edt if no missing----
    if mask.sum() == 0:
        slab = np.ones(img.shape)
        slab[seedy, seedx] = 0
        return distance_transform_edt(slab)

    target = (1 - mask).sum()
    dist = np.ones(img.shape) * np.inf
    dist[seedy, seedx] = 0

    def expandDir_1(img, direction):
        if direction == 'n':
            l1 = img[0, :]
            img = np.roll(img, 1, axis=0)
            img[0, :] == l1
        elif direction == 's':
            l1 = img[-1, :]
            img = np.roll(img, -1, axis=0)
            img[-1, :] == l1
        elif direction == 'e':
            l1 = img[:, 0]
            img = np.roll(img, 1, axis=1)
            img[:, 0] = l1
        elif direction == 'w':
            l1 = img[:, -1]
            img = np.roll(img, -1, axis=1)
            img[:, -1] == l1
        elif direction == 'ne':
            img = expandDir_1(img, 'n')
            img = expandDir_1(img, 'e')
        elif direction == 'nw':
            img = expandDir_1(img, 'n')
            img = expandDir_1(img, 'w')
        elif direction == 'sw':
            img = expandDir_1(img, 's')
            img = expandDir_1(img, 'w')
        elif direction == 'se':
            img = expandDir_1(img, 's')
            img = expandDir_1(img, 'e')

        return img

    def expandIter_1(img):
        sqrt2 = np.sqrt(2)
        tmps = []
        for dirii, dd in zip(['n', 's', 'e', 'w', 'ne', 'nw', 'sw', 'se'], [1, ] * 4 + [sqrt2, ] * 4):
            tmpii = expandDir_1(img, dirii) + dd
            tmpii = np.minimum(tmpii, img)
            tmps.append(tmpii)
        img = cv2.reduce(lambda x, y: np.minimum(x, y), tmps)

        return img

    # ----------------Iteratively expand----------------
    dist_old = dist
    while True:
        expand = expandIter_1(dist)
        dist = np.where(mask, dist, expand)
        nc = dist.size - len(np.where(dist == np.inf)[0])

        if nc >= target or np.all(dist_old == dist):
            break
        dist_old = dist
    return dist


def probability_map_calc_1(frame, background_mask, foreground_mask, y_width, x_width):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_chanel_density_fg, v_chanel_density_bg = pdf_estimation_1(hsv_frame, foreground_mask, background_mask, y_width, x_width)

    prob_given_fg = v_chanel_density_fg[hsv_frame[:, :, 1]]
    prob_given_bg = v_chanel_density_bg[hsv_frame[:, :, 1]]

    prob_fg = prob_given_fg / (prob_given_fg + prob_given_bg + 0.00000000000000002)
    prob_bg = 1 - prob_fg

    grad_bg = cv2.Sobel(prob_bg, cv2.CV_64F, 1, 1, ksize=5)
    grad_fg = cv2.Sobel(prob_fg, cv2.CV_64F, 1, 1, ksize=5)

    return grad_bg, grad_fg, prob_fg, prob_bg


def pdf_estimation_1(image, mask_fg, mask_bg, y_width, x_width):

    image = cv2.resize(image, (image.shape[0] // 4, image.shape[1] // 4))
    mask_fg = cv2.resize(mask_fg, (mask_fg.shape[0] // 4, mask_fg.shape[1] // 4))
    mask_bg = cv2.resize(mask_bg, (mask_bg.shape[0] // 4, mask_bg.shape[1] // 4))
    y_width = y_width//4
    x_width = x_width//4

    # find the center of the object
    rows_fg, cols_fg = np.where(mask_fg == 255)
    row_fg = int(np.mean(rows_fg))
    col_fg = int(np.mean(cols_fg))

    # crop the rectangular around the center

    image_crop = image[max(row_fg - y_width, 0): min(row_fg + y_width, mask_fg.shape[0] - 1),
                 max(col_fg - x_width, 0): min(col_fg + x_width, mask_fg.shape[1] - 1)]
    mask_crop_fg = mask_fg[max(row_fg - y_width, 0):min(row_fg + y_width, mask_fg.shape[0] - 1),
                   max(col_fg - x_width, 0):min(col_fg + x_width, mask_fg.shape[1] - 1)]
    mask_crop_bg = mask_bg[max(row_fg - y_width, 0):min(row_fg + y_width, mask_fg.shape[0] - 1),
                   max(col_fg - x_width, 0):min(col_fg + x_width, mask_fg.shape[1] - 1)]

    # calculate the kde of bg and fg
    color_level_grid = np.linspace(0, 255, 256)
    h, s, v = cv2.split(image_crop)
    rows, cols = np.where(mask_crop_fg == 255)
    info = s[rows, cols]
    try:
        pdf_fg = kde_scipy_1(info, color_level_grid)
    except:
        pdf_fg = mask_fg
    rows, cols = np.where(mask_crop_bg == 255)
    info = s[rows, cols]
    try:
        pdf_bg = kde_scipy_1(info, color_level_grid)
    except:
        pdf_bg = mask_bg
    return pdf_fg, pdf_bg


def kde_scipy_1(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def matting_func_1(background_name, stable_name, binary_name, alpha_name, trimap_name, matted_name, r=1.2, rho=10, y_width=600, x_width=200):
    """    main Function    """
    print("Stage 3: Video Matting")

    # set the parameters for the input and outputs videos
    image_vid = cv2.VideoCapture(stable_name)
    binary_vid = cv2.VideoCapture(binary_name)
    bg = cv2.imread(background_name)

    n_frames = int(image_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(image_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(image_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = image_vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out1 = cv2.VideoWriter(alpha_name, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(trimap_name, fourcc, fps, (width, height))
    out3 = cv2.VideoWriter(matted_name, fourcc, fps, (width, height))

    # small number
    eps = 0.000000000001

    i_f = 0
    for frame_n in tqdm(range(n_frames)):
        
        i_f += 1

        # read one frame
        ret1, img_frame = image_vid.read()
        ret2, binary_frame = binary_vid.read()
        if (ret1 * ret2 == 0):
            break

        binary_frame_gray = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)

        fg_mask = np.zeros_like(binary_frame_gray)
        bg_mask = np.zeros_like(binary_frame_gray)
        trimap = np.zeros_like(binary_frame_gray)
        alpha = np.zeros_like(binary_frame_gray)

        # calculate the fg and the bg based on the binary image
        fg_mask[binary_frame_gray >= 240] = 255
        bg_mask[binary_frame_gray <= 10] = 255
        # calculate the initial probability map and the geodesic distance
        grad_bg, grad_fg, prob_fg, prob_bg = probability_map_calc_1(img_frame, bg_mask, fg_mask, y_width, x_width)

        fg_gdf = geodesic_1(grad_fg, np.where(fg_mask == 255))
        bg_gdf = geodesic_1(grad_bg, np.where(bg_mask == 255))

        trimap[(fg_gdf - bg_gdf) > rho] = 0
        trimap[(bg_gdf - fg_gdf) > rho] = 255
        trimap[abs(bg_gdf - fg_gdf) <= rho] = 0.5 * 256

        # update the fg and the bg masks based on the trimap
        bg_mask = np.zeros_like(binary_frame_gray)
        fg_mask = np.zeros_like(binary_frame_gray)

        fg_mask[trimap == 255] = 255
        bg_mask[trimap == 0] = 255

        # calculate the Probability map based on the fg and bg masks and the geodesic distances
        grad_bg, grad_fg, prob_fg, prob_bg = probability_map_calc_1(img_frame, bg_mask, fg_mask, y_width, x_width)

        fg_gdf = geodesic_1(grad_fg, np.where(fg_mask == 255))
        bg_gdf = geodesic_1(grad_bg, np.where(bg_mask == 255))

        # update the trimap
        
        trimap[(fg_gdf - bg_gdf) >= rho] = 0
        trimap[(bg_gdf - fg_gdf) >= rho] = 255
        trimap[abs(bg_gdf - fg_gdf) < rho] = 0.5 * 256
        '''
        trimap[(fg_gdf - bg_gdf) >= rho] = 0
        trimap[(bg_gdf - fg_gdf) >= rho] = 1
        trimap[abs(bg_gdf - fg_gdf) < rho] = 0.5
        '''
        # calculate the weight based on the probability map and the geodesic distance
        W_fg = (fg_gdf + eps) ** (-r) * prob_fg
        W_bg = (bg_gdf + eps) ** (-r) * prob_bg

        # calc alpha map
        alpha = W_fg / (W_fg + W_bg)
        alpha[trimap == 255] = 1
        alpha[trimap == 0] = 0

        # Wrap the new background image to the frame
        frame = np.zeros_like(img_frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_frame_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        
        frame_hsv[:, :, 0] = alpha * img_frame_hsv[:, :, 0] + (1 - alpha) * bg_hsv[:, :, 0]
        frame_hsv[:, :, 1] = alpha * img_frame_hsv[:, :, 1] + (1 - alpha) * bg_hsv[:, :, 1]
        frame_hsv[:, :, 2] = alpha * img_frame_hsv[:, :, 2] + (1 - alpha) * bg_hsv[:, :, 2]
        '''
        frame_hsv[:, :, 0] = trimap * img_frame_hsv[:, :, 0] + (1 - trimap) * bg_hsv[:, :, 0]
        frame_hsv[:, :, 1] = trimap * img_frame_hsv[:, :, 1] + (1 - trimap) * bg_hsv[:, :, 1]
        frame_hsv[:, :, 2] = trimap * img_frame_hsv[:, :, 2] + (1 - trimap) * bg_hsv[:, :, 2]
        '''
        # Save the alpha, trimap and matted frame
        alpha = (alpha * 255).astype(np.uint8)

        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        trimap = cv2.cvtColor(trimap, cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

        out1.write(np.uint8(alpha))
        out2.write(np.uint8(trimap))
        out3.write(np.uint8(frame))

    # release all Videos
    out1.release()
    out2.release()
    out3.release()
    image_vid.release()
    binary_vid.release()
    cv2.destroyAllWindows()

