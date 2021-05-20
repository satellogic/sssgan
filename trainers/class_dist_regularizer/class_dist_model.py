import torch
import torch.nn as nn
from .models import create_modules, Darknet, EmptyLayer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def extend_modules(module_defs, module_list, new_module_defs):
    
    output_filters = [module_list[0][0].in_channels]
    create_idx = len(module_defs)
    for i, module_def in enumerate(module_defs + new_module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            filters = int(module_def['filters'])
            if i >= create_idx:   
                bn = int(module_def['batch_normalize'])
                
                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(module_def['stride']),
                                                            dilation=1,
                                                            padding=pad,
                                                            bias=not bn))

                if bn:
                    modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky_%d' % i, nn.LeakyReLU())

        elif module_def['type'] == 'upsample':
            if i >= create_idx:
                upsample = nn.Upsample(scale_factor=int(module_def['stride']))  # , mode='bilinear', align_corners=True)
                modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            if i >= create_idx:
                modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            if i >= create_idx:
                modules.add_module("shortcut_%d" % i, EmptyLayer())

        
        # Register module list and number of output filters
        if i >= create_idx: module_list.append(modules)
        output_filters.append(filters)

    return module_defs + new_module_defs, module_list





remove_modules_idx = [80,81,82,83, 92, 93, 94, 95, 104, 105, 106]
recover = [85, 97]

class ClassDistModel(nn.Module):
    def __init__(self, load=True, img_size=256, output_dim=10, freeze_backbone=False):
        super(ClassDistModel, self).__init__()
        self.output_dim = output_dim
        self.yolo_backbone_cfg = "/home/emilioty/ws/SPADE/trainers/class_dist_regularizer/cfg/c60_a30symmetric.cfg"
        model = Darknet(self.yolo_backbone_cfg, img_size)
        self.hyperparams = model.hyperparams

        if load:
            checkpoint = torch.load('/home/emilioty/ws/SPADE/trainers/class_dist_regularizer/weights/xview_best_lite.pt', map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        self._adapt_yolo_backbone(model)

        # if freeze_backbone:
        #     for param in self.module_list.parameters():
        #         param.requires_grad = False

    
    def _adapt_yolo_backbone(self, model):
        self.module_list = nn.ModuleList()
        self.module_defs = []
        scales_fm = []
        for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
            if i in remove_modules_idx:
                continue
            
            self.module_list.append(module)
            self.module_defs.append(module_def)

            if i in recover:
                scales_fm.append(len(self.module_defs) - 1)
        new_module_defs = []
        route = []   
        
        new_module_defs.append({"type":"route", "layers":str(scales_fm[0])})
        new_module_defs.append({"type": "convolutional",
                                "batch_normalize": "1",
                                "filters": "256",
                                "size": "1",
                                "stride": "1",
                                "pad": "1",
                                "activation": "leaky"})
        new_module_defs.append({"type":"upsample", "stride":"2"})    
        route.append(len(new_module_defs)-1 + len(self.module_defs))

        new_module_defs.append({"type":"route", "layers":str(scales_fm[1])})
        new_module_defs.append({"type": "convolutional",
                                "batch_normalize": "1",
                                "filters": "256",
                                "size": "1",
                                "stride": "1",
                                "pad": "1",
                                "activation": "leaky"})   
        route.append(len(new_module_defs)-1 + len(self.module_defs))

        route.append(len(self.module_defs)-1)
        new_module_defs.append({"type":"route", "layers":",".join([str(r) for r in route])})
        new_module_defs.append({"type": "convolutional",
                                    "batch_normalize": "1",
                                    "filters": "512",
                                    "size": "1",
                                    "stride": "1",
                                    "pad": "1",
                                    "activation": "leaky"})
        self.module_defs, self.module_list = extend_modules(self.module_defs, self.module_list, new_module_defs)

        self.output_module = nn.ModuleList()
        self.output_module.append(nn.MaxPool2d(2))

        conv_output = nn.Sequential()
        conv_output.add_module('conv_out_1', nn.Conv2d(in_channels=512,
                                                        out_channels=512,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        bias=False))

        
        conv_output.add_module('batch_norm_conv_out_1', nn.BatchNorm2d(512))
        conv_output.add_module('leaky_conv_out_1', nn.LeakyReLU())
        self.output_module.append(conv_output)
        conv_output = nn.Sequential()
        conv_output.add_module('conv_out_2', nn.Conv2d(in_channels=512,
                                                        out_channels=256,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        bias=False))

        
        conv_output.add_module('batch_norm_conv_out_2', nn.BatchNorm2d(256))
        conv_output.add_module('leaky_conv_out_2', nn.LeakyReLU())
        conv_output.add_module('conv_out_3', nn.Conv2d(in_channels=256,
                                                        out_channels=256,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        bias=False))

        
        conv_output.add_module('batch_norm_conv_out_3', nn.BatchNorm2d(256))
        conv_output.add_module('leaky_conv_out_3', nn.LeakyReLU())
        self.output_module.append(conv_output)
        self.output_module.append(nn.MaxPool2d(2))
        self.output_module.append(Flatten())
        self.output_module.append(nn.Linear(256*8*8, 2048))
        self.output_module.append(nn.LeakyReLU())
        self.output_module.append(nn.Linear(2048, 2048))
        self.output_module.append(nn.LeakyReLU())
        self.output_module.append(nn.Linear(2048, 1024))
        self.output_module.append(nn.LeakyReLU())
        self.output_module.append(nn.Linear(1024, self.output_dim))


    def forward(self, x):
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            
            layer_outputs.append(x)

        for module in self.output_module:
            x = module(x)

        return x