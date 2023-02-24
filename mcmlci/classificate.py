import json
import argparse

import torch

from mcmlic_model import MultiChannelMultiLabelImageCrassifer

def classificate(opt):
    # classificate multi label from multi channel image
    device = torch.device(opt.device)
    # with open(opt.conf,'r') as f:
    #     conf = json.load(f)
    #     input_channels = conf['input_channels']
    #     output_labels = conf['output_labels']
    input_channels = opt.input_channels_list
    output_labels = opt.output_labels_list

    print('input_channels_data:', input_channels)
    print('output_labels_data:', output_labels)
    model = MultiChannelMultiLabelImageCrassifer(num_input_channels=len(input_channels),num_output_labels=len(output_labels)).to(device)
    print('MCMLIC model loading...')
    model.load_state_dict(torch.load(opt.model, map_location=device))
    model = model.eval()

    if type(input) == str:  # input is file path
        input = torch.load(opt.input, map_location=device)
    elif type(input) == torch.Tensor: # input is torch.Tensor
        pass
    else:
        assert False, 'mcmlic input type error'

    input = input.unsqueeze(0)
    logit = model(input)  # Predict result(float)
    probs = torch.sigmoid(logit) # sigmoid result(from 0 to 1)
    predictions = (probs >= opt.threshold).float()  # Thresholding with 0.5
    predictions = torch.Tensor.cpu(predictions).detach().numpy()[0] #result [1,1,1,0,0,1,0,,,]
    predicted_labels = [x for i, x in enumerate(output_labels) if predictions[i] == 1]

    return predicted_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi channel multi label image crassifer prediction')
    parser.add_argument('--input',required=True, help='.pt file ,torch type')
    parser.add_argument('--model', required=True)
    # parser.add_argument('--conf', required=True, help='json file')
    parser.add_argument('--input_channels_list', required=True, help='yolo classes')
    parser.add_argument('--output_labels_list', required=True)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--device',default='cuda')
    opt = parser.parse_args()
    predicted_labels = classificate(opt)
    print(predicted_labels)

    # / yolo2vgg / data / mcmlic_conf.json
    # / home / solution - 2020 - 2 / PycharmProjects / yolov7 - main / yolo2vgg / test1 / image_set / 2199.pt
