import cv2


def image_blending(value_Img_dir, save_dir, value_Img_half_dir, half_save_dir):
    value_Img = cv2.imread(
        value_Img_dir)
    value_Img_half = cv2.imread(
        value_Img_half_dir)
    background = cv2.imread("../resource/hockey-field.png")
    # v_rows, v_cols, v_channels = value_Img.shape
    # v_h_rows, v_h_cols, v_h_channels = value_Img_half.shape

    focus_Img = value_Img[60:540, 188:1118]
    f_rows, f_cols, f_channels = focus_Img.shape
    focus_background = cv2.resize(background, (f_cols, f_rows), interpolation=cv2.INTER_CUBIC)
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 0.5, -255 / 2)
    blend_all = value_Img
    blend_all[60:540, 188:1118] = blend_focus
    # final_rows = v_rows * float(b_rows) / float(f_rows)
    # final_cols = v_cols * float(b_cols) / float(f_cols)
    # blend_all_final = cv2.resize(blend_all, (int(final_cols), int(final_rows)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('res', focus_Img)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir, blend_all)

    focus_Img_half = value_Img_half[120:1090, 190:1125]
    f_h_rows, f_h_cols, f_h_channels = focus_Img_half.shape
    focus_background_half = cv2.resize(background[:, 899:1798, :], (f_h_cols, f_h_rows), interpolation=cv2.INTER_CUBIC)
    blend_half_focus = cv2.addWeighted(focus_Img_half, 1, focus_background_half, 0.5, -255 / 2)
    blend_half_all = value_Img_half
    blend_half_all[120:1090, 190:1125] = blend_half_focus
    cv2.imwrite(half_save_dir, blend_half_all)
