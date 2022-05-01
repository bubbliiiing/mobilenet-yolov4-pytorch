#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape     = [416, 416]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    backbone        = 'mobilenetv1'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(anchors_mask, num_classes, backbone=backbone).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    
    # mobilenetv1-yolov4 40,952,893
    # mobilenetv2-yolov4 39,062,013
    # mobilenetv3-yolov4 39,989,933

    # 修改了panet的mobilenetv1-yolov4 12,692,029
    # 修改了panet的mobilenetv2-yolov4 10,801,149
    # 修改了panet的mobilenetv3-yolov4 11,729,069

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    