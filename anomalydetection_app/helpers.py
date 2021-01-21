import numpy as np
from anomalydetection import detectors


def instantiate_detector_instance(selected_detector):
    detector_presence = [selected_detector == detector_option['value'] for detector_option in get_anomaly_detectors()]
    detector_index = np.where(detector_presence)[0][0]
    current_detector = getattr(detectors, get_anomaly_detectors()[detector_index]['value'])()
    return current_detector


def list_contains_value_in_dict(list_of_dicts, key_to_inspect, value_sought):
    valid_items = [item for item in list_of_dicts if item[key_to_inspect] == value_sought]
    return len(valid_items) > 0


def checklist_option_from_object(detector_class_name):
    return {'label': detector_class_name, 'value': detector_class_name}


def get_anomaly_detectors():
    anomaly_detectors = [checklist_option_from_object(detector) for detector in dir(detectors)
                         if ('Detector' in detector) and ('Base' not in detector)]
    return anomaly_detectors
