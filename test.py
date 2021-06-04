import torch
from torchsummary import summary

from nets.yolo4 import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = YoloBody(3,80,backbone="ghostnet").to(device)
    summary(model, input_size=(3, 416, 416))
    
    # mobilenetv1-yolov4 40,952,893
    # mobilenetv2-yolov4 39,062,013
    # mobilenetv3-yolov4 39,989,933

    # 修改了panet的mobilenetv1-yolov4 12,692,029
    # 修改了panet的mobilenetv2-yolov4 10,801,149
    # 修改了panet的mobilenetv3-yolov4 11,729,069
