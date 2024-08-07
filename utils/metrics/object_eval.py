from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

def imagenet_ResNet50(device):
    processor = AutoImageProcessor.from_pretrained('/vhome/gongchao/.cache/huggingface/hub/models--microsoft--resnet-50/snapshots/4067a2728b9c93fbd67b9d5a30b03495ac74a46e', cache_dir=".cache")
    model = ResNetForImageClassification.from_pretrained("/vhome/gongchao/.cache/huggingface/hub/models--microsoft--resnet-50/snapshots/4067a2728b9c93fbd67b9d5a30b03495ac74a46e", cache_dir=".cache")
    model.to(device)
    return processor, model

def object_eval(classifier, img, processor, device):
    with torch.no_grad():
        inputs = processor(img, return_tensors="pt")
        inputs.to(device)
        logits = classifier(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    # print(predicted_label)
    # print(classifier.config.id2label[predicted_label])
    return predicted_label, torch.softmax(logits, dim=-1).squeeze()