import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

def draw_color_bar(ndarray, color_bar_width=1500, color_bar_height=30, color_flag='green'):
    """Return ndarray color bar image.
    Args:
        ndarray: one dimension ndarray of boolean.
        color_bar_width: int
        color_bar_height: int
        color_flag: int
    Returns:
        ndarray: color bar image.
    """
    # check ndarray size
    if ndarray.size > color_bar_width:
        color_bar_width = ndarray.size

    # initialize color bar image
    image = np.zeros((color_bar_height, color_bar_width, 3), dtype=np.uint8)

    # calculate grid for each ndarray element
    grid = np.linspace(0, color_bar_width, color_bar_width + 1).astype(int)

    # create a mask from ndarray elements０
    mask = ndarray.astype(bool)

    # draw color bar as rectangle corresponding to ndarray value
    y1, y2 = 0, color_bar_height

    # set color based on color_flag
    if color_flag == "green":
        color = (0, 255, 0)
    elif color_flag == "white":
        color = (255, 255, 255)
    elif color_flag == "red":
        color = (0, 0, 255)

    # draw rectangle for each masked x1 and x2
    rectangles = []
    rect = None
    for i, val in enumerate(mask):
        if val:
            x1 = grid[-(i + 2)]
            x2 = grid[-(i + 1)]
            print('check rect:',rect, x1,x2)
            if not rect:
                # set new rect
                rect = (x1, x2)
            elif rect and x2 == rect[0]:
                # merge rect
                rect = (x1, rect[1])
            elif rect and x2 != rect[0]:
                # add rect
                rectangles.append(rect)
                # set new rect
                rect = (x1, x2)

    if rect:
        # add last rect
        rectangles.append(rect)
        del rect

    for r in rectangles:
        cv2.rectangle(image, (r[0], y1), (r[1], y2), color, -1)

    # draw outside rectangle
    cv2.rectangle(image, (0, 0), (color_bar_width, color_bar_height), (0, 0, 0), 1)

    return image

def draw_labels_in_rectangle(str, rectangle_width,rectangle_height, font_size=1.5, font_color=(255, 255, 255)):
    """Return ndarray image with labels in rectangle.
    Args:
        list: list of string.
        rectangle_width: int
        rectangle_height: int
        font_size: float
        font_color: tuple
    Returns:
        ndarray: image with labels in rectangle.
    """
    # initialize image
    image = np.zeros((rectangle_height, rectangle_width, 3), dtype=np.uint8)

    # draw rectangle
    cv2.rectangle(image, (0, 0), (rectangle_width, rectangle_height), (0, 0, 0), -1)

    # draw label
    cv2.putText(image, str, (5, 22), cv2.FONT_HERSHEY_PLAIN, font_size, font_color, 1)

    # draw outside rectangle
    cv2.rectangle(image, (0, 0), (rectangle_width, rectangle_height), (255, 255, 255), 1)

    return image

label = "sample text ////////////////"
image = draw_labels_in_rectangle(label, 300, 30)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def draw_prediction_time_chart(predictions_recode,label_list, total_width=1800, table_width=300,
                               chart_width = 1500,font_size=1.5, font_color=(255, 255, 255)):
    """Return ndarray image with prediction time chart.
    Args:
        predictions_recode: ndarray. [ frame_num, label_num ]
        label_list: list of string.
        total_width: int
        total_height: int
        table_width: int
        chart_width: int
        font_size: float
        font_color: tuple
    Returns:
        ndarray: image with prediction time chart.
    """

    total_height = 30 * len(label_list)

    # initialize image
    image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # draw yolo image
    # resize yolo image to (total_height - , total_width)

    # draw table(labels_in_rectangle) from label_list
    # rectangles are aligned vertically
    rectangle_width = table_width
    rectangle_height = 30
    for i, label in enumerate(label_list):
        image[rectangle_height*i:rectangle_height*(i+1), 0:rectangle_width] = \
            draw_labels_in_rectangle(label, rectangle_width, rectangle_height, font_size, font_color)

    # draw chart from predictions_recode with color bar
    # color bar is aligned vertically
    color_bar_width = chart_width
    color_bar_height = 30

    # transpose predictions_recode (frame_num, class_num) -> (class_num, frame_num)
    predictions_recode = predictions_recode.transpose()

    # draw color bar
    for i, prediction in enumerate(predictions_recode):
        # check latest prediction status to determine color flag
        # if latest prediction is True and recent specified number of predictions are True,draw red color bar.
        # if latest prediction is True , draw green color bar.
        # if latest prediction is False, draw white color bar.

        if prediction[-1] == True:
            if np.sum(prediction[-100:]) == 100:
                color_flag = "red"
            else:
                color_flag = "green"
        else:
            color_flag = "white"

        # draw color bar
        # check need to resize color bar or not
        if predictions_recode.shape[1] > color_bar_width:
            color_bar = draw_color_bar(prediction, chart_width, color_bar_height, color_flag='green')
            color_bar = cv2.resize(color_bar, (chart_width, color_bar_height))
            image[color_bar_height*i:color_bar_height*(i+1), rectangle_width:total_width] = color_bar

        else:
            color_bar = draw_color_bar(prediction, chart_width, color_bar_height, color_flag='green')
            image[color_bar_height*i:color_bar_height*(i+1), rectangle_width:total_width] = color_bar

    return image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def cv2_putText_2(img, text, org, fontFace, fontScale, color):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    w, h = draw.textsize(text, font = fontPIL)
    draw.text(xy = (x,y-h), text = text, fill = colorRGB, font = fontPIL)
    imgCV = pil2cv(imgPIL)
    return imgCV