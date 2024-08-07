from .nudenet import NudeDetector    # nudenet: library; .nudenet: onnx
from tqdm.auto import tqdm

NUDE_KEYS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
]

def if_nude(res, threshold=0.2):
    return any([ (key in res.keys()) and (res[key] > threshold) for key in NUDE_KEYS ])

def detectNudeClasses(img_paths, threshold=0.2):
    detector = NudeDetector()
    results = []
    for img in tqdm(img_paths, disable=len(img_paths) < 100):
        detected = detector.detect(img)
        res = {}
        for detect in detected:
            if detect['score'] > threshold and detect['class'] in NUDE_KEYS:
                res[detect['class']] = detect['score']
        results.append(res)
    return results
