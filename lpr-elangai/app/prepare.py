"""
Preparation of lpr-elangai-trt application.

Generated by 'elangai generate' using `elangai` 1.1.0.
"""


from elangai import *


vehicle_model = 'vehicle.trt' # Only accepts '.trt' or '.tflite'
vehicle_label = "vehicle_labels.txt" # if not available put an empty string ("")
InferencePool(vehicle_model, vehicle_label, FP16)

# plate_model = "plate.trt"
# plate_label = ""
# InferencePool(plate_model, plate_label, FP16)
