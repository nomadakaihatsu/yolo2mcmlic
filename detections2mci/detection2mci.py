'''
class purpose
this class is used to convert yolo detections result to mcmlic input.

input / output
input: detections result N * [x1,y1,x2,y2,conf,class] as torh.Tensor
output: mcmlic input is torh.Tensor, its shape is [1, num of yolo classes, height=224, width=224]

class outline
1. initialize mcmlic data by torch.zeros.
2. recode yolo detections result in frames with specified frame number.
3. generate mcmlic data by recorded yolo detections result.
4. generate demo view by mcmlic data.

'''

import os
import torch
import numpy as np
import cv2

# classifer settings
CLASSIFIER_TYPE = '2DConv'
MCI_METHOD = 'WMA'
NUM_OF_FRAMES = 30

# demo view settings
DEMO_X = 960
DEMO_Y = 960
DEMO_X_GRID = 5
DEMO_Y_GRID = 3


class Detections2Mci():
    def __init__(self, name, yolo_labels, yolo_x, yolo_y, yolo_fps, mci_x, mci_y, classifier_type=CLASSIFIER_TYPE,
                 mci_method=MCI_METHOD,
                 num_of_frames=NUM_OF_FRAMES, demo_x=DEMO_X, demo_y=DEMO_Y, demo_x_grid=DEMO_X_GRID,
                 demo_y_grid=DEMO_Y_GRID,
                 save_mci=False, show_demo_view=False, save_demo_view=False):
        self.device = torch.device('cuda')
        self.name = name
        self.classifier_type = classifier_type
        self.mci_method = mci_method
        self.num_of_frames = num_of_frames
        self.yolo_labels = yolo_labels
        self.yolo_x = yolo_x
        self.yolo_y = yolo_y
        self.yolo_fps = yolo_fps
        self.mci_x = mci_x
        self.mci_y = mci_y
        self.demo_x = demo_x
        self.demo_y = demo_y
        self.demo_x_grid = demo_x_grid
        self.demo_y_grid = demo_y_grid
        self.mci = torch.zeros((1, len(self.yolo_labels), self.mci_y, self.mci_x)).to(self.device)
        self.save_mci = save_mci
        self.show_demo_view = show_demo_view
        self.save_demo_view = save_demo_view
        if classifier_type == '2DConv':
            self.yolo_recode = []
        elif classifier_type == '3Dconv':
            assert False, 'not implemented'
        else:
            assert False, 'classifier error'
        if save_mci:
            # save mci in runs/mci/name directory, if directory is not exist, create it.
            if not os.path.exists('runs/mci/' + self.name):
                os.makedirs('runs/mci/' + self.name)
        if save_demo_view:
            # save demo_view in runs/demo directory, if directory is not exist, create it.
            if not os.path.exists('runs/demo/' + self.name):
                os.makedirs('runs/demo/' + self.name)
            # initialize demo_view_writer
            self.demo_view_writer = cv2.VideoWriter('runs/demo/' + self.name + '.avi',
                                                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 1,
                                                    (self.demo_x, self.demo_y))

    def generate_mci(self, yolo_detections,frame_number, yolo_img=None):
        if frame_number % self.yolo_fps != 0:
            if self.classifier == '2DConv':
                self.yolo_recode.append(yolo_detections)

                # remove the oldest frame
                if len(self.yolo_recode) > self.num_of_frames:
                    self.yolo_recode.pop(0)

                # update mci by weighted moving average method
                if self.mci_method == 'WMA':
                    self.mci = torch.zeros((1, len(self.yolo_labels), self.mci_y, self.mci_x)).to(self.device)
                    for i, yolo_detections in enumerate(reversed(self.yolo_recode)):
                        if len(yolo_detections) == 0:
                            # if no detection in this frame,
                            continue
                        for yolo_detection in yolo_detections:
                            yolo_x1 = yolo_detection[0]
                            yolo_y1 = yolo_detection[1]
                            yolo_x2 = yolo_detection[2]
                            yolo_y2 = yolo_detection[3]
                            yolo_label = yolo_detection[5]
                            mci_x1 = int(yolo_x1 / self.yolo_img_x * self.mci_x)
                            mci_y1 = int(yolo_y1 / self.yolo_img_y * self.mci_y)
                            mci_x2 = int(yolo_x2 / self.yolo_img_x * self.mci_x)
                            mci_y2 = int(yolo_y2 / self.yolo_img_y * self.mci_y)
                            self.mci[0, yolo_label, mci_y1:mci_y2, mci_x1:mci_x2] += 1 * (len(self.yolo_recode) - 1)

                    # normalize with considering overlapping detections area
                    self.mci = self.mci / (sum(range(len(self.yolo_recode))) * 2)

                else:
                    assert False, 'mci_method error'

            elif self.classifier == '3DConv':
                assert False, 'not implemented'
            else:
                assert False, 'classifier error'

            # save mci in .pt file in runs/mci directory
            # the file name is frame_id.pt
            if self.save_mci:
                dataset_id = frame_number // self.yolo_fps
                torch.save(self.mci, 'runs/mci/' + self.name + '/' + str(dataset_id) + '.pt')

            if self.show_demo_views or self.save_demo_views:
                # generate demo view
                demo_view = self.generate_demo_view(yolo_img)

                # show demo view
                if self.show_demo_view:
                    cv2.imshow('demo_view', demo_view)
                    cv2.waitKey(1)

                # save demo view
                if self.save_demo_view:
                    self.demo_view_writer.write(demo_view)

    def get_mci(self):
        return self.mci

    def generate_demo_view(self, yolo_img):
        '''
        demo view is a view of mci for visualization.
        the size of demo view is specified by demo_x and demo_y.
        demo view is created by compositing the original video and the grayscale image from each channel of mci.
        the original video is generarted from yolo detections result with RGB color.
        the grayscale images are generated from each channel of mci with grayscale color.
        the original video is composited in the top side of demo view.
        the grayscale images are composited in the bottom side of demo view.
        the grayscale images are arranged in a grid according to demo_x_grid and demo_y_grid.
        the size of each grayscale image is calculated by demo_x and demo_y and demo_x_grid and demo_y_grid.
        the order of grayscale images is from left to right and from top to bottom.
        the order of grayscale images is the same as the order of yolo labels.
        '''
        # create demo view
        demo_view = np.zeros((self.demo_y, self.demo_x, 3), dtype=np.uint8)
        # composite original video
        yolo_img = cv2.resize(yolo_img, (self.demo_x, self.demo_y // 2))
        demo_view[:self.demo_y // 2, :, :] = yolo_img
        # composite grayscale images
        mci = self.mci[0].cpu().detach().numpy()
        mci = mci * 255
        mci = mci.astype(np.uint8)

        # calculate the size of each grayscale image
        demo_x_grid_size = self.demo_x // self.demo_x_grid
        demo_y_grid_size = (self.demo_y // 2) // self.demo_y_grid
        for i in range(len(self.yolo_labels)):
            # calculate the position of each grayscale image
            demo_x_grid_pos = i % self.demo_x_grid
            demo_y_grid_pos = i // self.demo_x_grid
            demo_x_pos = demo_x_grid_pos * demo_x_grid_size
            demo_y_pos = demo_y_grid_pos * demo_y_grid_size + self.demo_y // 2
            # composite each grayscale image
            mci_img = cv2.resize(mci[i], (demo_x_grid_size, demo_y_grid_size))
            mci_img = cv2.cvtColor(mci_img, cv2.COLOR_GRAY2BGR)
            demo_view[demo_y_pos:demo_y_pos + demo_y_grid_size, demo_x_pos:demo_x_pos + demo_x_grid_size, :] = mci_img

        # draw grid lines in demo view
        for i in range(self.demo_x_grid):
            if i == 0:
                continue
            else:
                demo_view[:, i * demo_x_grid_size, :] = (255, 255, 255)
        for i in range(self.demo_y_grid):
            demo_view[i * demo_y_grid_size + self.demo_y // 2, :, :] = (255, 255, 255)

        # put yolo labels in demo view
        # if none label in the grid cell, draw 'None' label
        for i in range(len(self.yolo_labels)):
            # calculate the position of each yolo label
            demo_x_grid_pos = i % self.demo_x_grid
            demo_y_grid_pos = i // self.demo_x_grid
            demo_x_pos = demo_x_grid_pos * demo_x_grid_size
            demo_y_pos = demo_y_grid_pos * demo_y_grid_size + self.demo_y // 2
            if i < len(self.yolo_labels) - 1:
                # put each yolo label
                cv2.putText(demo_view, self.yolo_labels[i], (demo_x_pos, demo_y_pos + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # put 'Provision' label
                cv2.putText(demo_view, 'Provision', (demo_x_pos, demo_y_pos + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return demo_view

    def __del__(self):
        if self.show_demo_view:
            cv2.destroyAllWindows()
        if self.save_demo_view:
            self.demo_view_writer.release()
