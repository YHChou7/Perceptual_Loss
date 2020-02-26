import torch
import torch.nn as nn 
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel = 9, stride = 1)
        self.in1 = nn.InstanceNorm2d(32, affine = True)
        self.conv2 = ConvLayer(32, 64, kernel = 3, stride = 2)
        self.in2 = nn.InstanceNorm2d(64, affine = True)
        self.conv3 = ConvLayer(64, 128, kernel = 3, stride = 2)
        self.in3 = nn.InstanceNorm2d(128, affine = True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsample layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel = 3, stride =1, upsample = 2)
        self.in4 = nn.InstanceNorm2d(64, affine = True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel = 3, stride = 1, upsample = 2)
        self.in5 = nn.InstanceNorm2d(32, affine = True)
        self.deconv3 = ConvLayer(32, 3, kernel = 9, stride = 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.relu(self.in4(self.deconv1(out)))
        out = self.relu(self.in5(self.deconv2(out)))
        out = self.deconv3(out)
        return out



class ConvLayer(nn.Module):
    def __init__(self, input, output, kernel, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel // 2
        self.reflection_padding = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(input, output, kernel, stride)
    
    def forward(self, x):
        out = self.reflection_padding(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channel, channel, kernel=3, stride =1)
        self.instnace_norm1 = nn.InstanceNorm2d(channel, affine = True)
        self.conv2 = ConvLayer(channel, channel, kernel=3, stride =1)
        self.instnace_norm2 = nn.InstanceNorm2d(channel, affine = True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.instnace_norm1(self.conv1(x)))
        out = self.instnace_norm2(self.conv2(out))
        out = residual + out
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, input, output, kernel, stride, upsample = None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel // 2
        self.reflection_padding = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(input, output, kernel, stride)

    def forward(self, x):
        x_in = x 
        if self.upsample :
            x_in = nn.functional.interpolate(x_in, mode = "nearest", scale_factor = self.upsample)
        out = self.reflection_padding(x_in)
        out = self.conv2d(out)
        return out
