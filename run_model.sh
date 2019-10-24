#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh
python3 runway_model.py -m models/squeezenet/1/squeezenet1.1.xml
