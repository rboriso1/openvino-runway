# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#import random
from __future__ import print_function
import sys
import os
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore
from glob import glob

class ExampleModel():

    def __init__(self, options):
        try:
            model_xml = glob(options['model_directory'] + '/*.xml')[0]
        except:
            raise Exception('Could not find .xml file in the specified directory')

        try:
            model_bin = glob(options['model_directory'] + '/*.bin')[0]
        except:
            raise Exception('Could not find .bin file in the specified directory')

        try:
            model_labels = glob(options['model_directory'] + '/*.labels')[0]
        except:
            print('Could not find .labels file, skipping labels')

        # Plugin initialization for specified device and load extensions library if specified
        print("Creating Inference Engine")
        ie = IECore()
        #if args.cpu_extension and 'CPU' in args.device:
        #    ie.add_extension(args.cpu_extension, "CPU")
        # Read IR
        print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        self.net = IENetwork(model=model_xml, weights=model_bin)

        if options['device'] == 'CPU':
            supported_layers = ie.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                print("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(options['device'], ', '.join(not_supported_layers)))
                print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        assert len(self.net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample supports only single output topologies"

        print("Preparing input blobs")
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1 #len(args.input)

        self.labels = model_labels
        self.number_top = options['number_top']

        # Loading model to the plugin
        print("Loading model to the plugin")
        self.exec_net = ie.load_network(network=self.net, device_name=options['device'])
        print("Model loaded")

    # Generate a label based on image.
    def run_on_input(self, in_image):
        n, c, h, w = self.net.inputs[self.input_blob].shape
        print("Batch size is {}".format(n))
        images = np.ndarray(shape=(n, c, h, w))
        for i in range(n):
            image = np.array(in_image)
            if image.shape[:-1] != (h, w):
                print("Image is resized from {} to {}".format(image.shape[:-1], (h, w)))
                image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[i] = image

        # Start sync inference
        print("Starting inference in synchronous mode")
        res = self.exec_net.infer(inputs={self.input_blob: images})

        # Processing output blob
        print("Processing output blob")
        res = res[self.out_blob]
        print("Top {} results: ".format(self.number_top))
        if self.labels:
            with open(self.labels, 'r') as f:
                labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        else:
            labels_map = None
        classid_str = "classid"
        probability_str = "probability"
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-self.number_top:][::-1]
            result = "OpenVINO Classification Sample\n\n"
            result += "{}\t{}\n".format(classid_str, probability_str)
            result += "{}\t{}\n".format('-' * len(classid_str), '-' * len(probability_str))
            for id in top_ind:
                det_label = labels_map[id] if labels_map else "{}".format(id)
                label_length = len(det_label)
                space_num_before = (len(classid_str) - label_length) // 2
                space_num_after = len(classid_str) - (space_num_before + label_length) + 2
                space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
                result += "{}{}{}{}{:.7f}\n".format(' ' * space_num_before, det_label,
                                              ' ' * space_num_after, ' ' * space_num_before_prob,
                                              probs[id])
        print("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")
        return result
