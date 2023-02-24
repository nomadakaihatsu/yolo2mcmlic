'''
scripy purpose
this script predict work label from yolo detection result.

input / output
input: video, config file, work record file(option)
output: work predict result below
    1. work time chart (csv) from predict result
    2. work time chart (csv) from yolo detection result(option)
    3. work time chart (csv) from work record file(option)
    # if above option are selected, option time chart is attached to work time chart from predict result
    4. video output with yolo detections(BB) and embed work label from predict result

script outline
    1. load config file, video file and work record file(option).
    2. predict yolo detection result from video file.
    3. convert yolo detection result to multi-channel(channel number is yolo detections class number) image.
    4. predict work label from multi-channel image.
    5. output above.

'''
import time
import random
from pathlib import Path
import json
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2

from yolov7.detect import detect
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from detections2mci.detection2mci import Detections2Mci
from mcmlci.mcmlic_model import MultiChannelMultiLabelImageCrassifer
from mcmlci.view_tool import draw_prediction_time_chart


def detect_to_classificate(opt):
    # load command line arguments
    source = opt.source
    name = opt.name
    detect_to_classificate_conf_path = opt.detect_to_classificate_conf_path
    # load config file
    with open(detect_to_classificate_conf_path, 'r') as f:
        detect_to_classificate_conf = json.load(f)
    yolo_labels = detect_to_classificate_conf['yolo_labels']
    work_labels = detect_to_classificate_conf['work_labels']
    classifier_type = detect_to_classificate_conf['classifier_type']
    mci_method = detect_to_classificate_conf['mci_method']
    num_of_frames = detect_to_classificate_conf['num_of_frames']
    mci_x = detect_to_classificate_conf['mci_x']
    mci_y = detect_to_classificate_conf['mci_y']
    conf_thred = detect_to_classificate_conf['conf_thred']
    preview_x = detect_to_classificate_conf['preview_x']
    preview_y = detect_to_classificate_conf['preview_y']
    preview_yolo_x = detect_to_classificate_conf['preview_yolo_x']
    preview_yolo_y = detect_to_classificate_conf['preview_yolo_y']
    preview_table_x = detect_to_classificate_conf['preview_table_x']
    preview_chart_x = detect_to_classificate_conf['preview_chart_x']
    preview_chart_height = detect_to_classificate_conf['preview_chart_height']
    show_preview = detect_to_classificate_conf['show_preview']
    save_preview = detect_to_classificate_conf['save_preview']
    mcmlci_model = detect_to_classificate_conf['mcmlci_model']
    # yolo config
    weights = detect_to_classificate_conf['yolo_argument']['weights']
    imgsz = detect_to_classificate_conf['yolo_argument']['imgsz']
    conf_thres = detect_to_classificate_conf['yolo_argument']['conf_thres']
    iou_thres = detect_to_classificate_conf['yolo_argument']['iou_thres']
    device = detect_to_classificate_conf['yolo_argument']['device']

    # save path setup
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    # yolov7 setup
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolo7 model
    print("load yolo model..")
    yolo7_model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(yolo7_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = yolo7_model.module.names if hasattr(yolo7_model, 'module') else yolo7_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        yolo7_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolo7_model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    # mcmlic setup
    # refer to first dataset to get yolo_x, yolo_y and fps
    _, _, _, vid_cap = next(iter(dataset))
    yolo_x = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yolo_y = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    yolo_fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    # setup detections2mci converter
    detections2mci = Detections2Mci(name, yolo_labels, yolo_x, yolo_y, yolo_fps, mci_x, mci_y,
                                    classifier_type, mci_method, num_of_frames)

    # setup mcmlci model
    print("setup mcmlci model..")
    mcmlci_model = MultiChannelMultiLabelImageCrassifer(num_input_channels=len(yolo_labels),
                                                        num_output_channels=len(work_labels))
    mcmlci_model.to(device)
    print("load mcmlci model..")
    mcmlci_model.load_state_dict(torch.load(mcmlci_model, map_location=device))
    mcmlci_model = mcmlci_model.eval()

    # initialize mci prediction record
    print("initialize mci prediction record..")
    predictions_record = []

    # load dataset
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                yolo7_model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = yolo7_model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                if show_preview or save_preview:  # Add bbox to image
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            ### mcmlic process start
            # get current frame number
            frame_count = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
            detections2mci.generate_mci(yolo_detections=det, yolo_img=im0, frame_count=frame_count)
            mci = detections2mci.get_mci()
            mci = mci.unsqueeze(0)
            logit = mcmlci_model(mci)
            probs = torch.sigmoid(logit)
            predictions = (probs > 0.5).float()
            predictions = torch.Tensor.cpu(predictions).detach().numpy()[0]
            predicted_labels = [x for i, x in enumerate(work_labels) if predictions[i] == 1]
            print("frame: ", frame_count," predicted labels: ", predicted_labels)
            # add predictions to predictions record
            # TODO: below process are implemented in MciRecodeHandler class
            if frame_count % yolo_fps == 0:
                predictions_record.append(predicted_labels)
                # if length of predictions_record is greater than num_of_frames, remove the first element
                if len(predictions_record) > num_of_frames:
                    predictions_record.pop(0)

            # generate preview
            if show_preview or save_preview:
                preview_img = np.zeros(preview_y, preview_x, 3)
                yolo_viwe = cv2.resize(im0, (preview_yolo_x, preview_yolo_y))
                yolo_offset_x = int((preview_x - preview_yolo_x) / 2)
                yolo_offset_y = 50
                preview_img[yolo_offset_y:yolo_offset_y + preview_yolo_y,
                yolo_offset_x:yolo_offset_x + preview_yolo_x] = yolo_viwe
                time_chart = draw_prediction_time_chart(predictions_record, work_labels,
                                                        preview_chart_x + preview_table_x,
                                                        preview_table_x, preview_chart_x)
                chart_offset_x = int((preview_x - (preview_table_x + preview_chart_x) / 2))
                chart_offset_y = int(yolo_offset_y + preview_yolo_y + 50)
                preview_chart_height = preview_chart_height * len(work_labels)
                preview_img[chart_offset_y:chart_offset_y + preview_chart_height,
                chart_offset_x:chart_offset_x + preview_chart_x + preview_table_x] = time_chart

            if show_preview:
                cv2.imshow('preview', preview_img)
                cv2.waitKey(1)

            if save_preview:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = 1
                    else:  # stream
                        fps = 1
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (preview_x, preview_y))
                vid_writer.write(preview_img)

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    # load command line arguments
    parser = argparse.ArgumentParser()
    # argument not for yolov7 is from command line
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--detect_to_classificate_conf_path', type=str, default='conf/detect_to_classificate_conf.json')
    parser.add_argument('--project', default='runs/predict', help='save results to project/name')
    parser.add_argument('--names', type=str, default='exp', help='*.names path')
    opt = parser.parse_args()
    print(opt)
    detect_to_classificate(opt)
