from .q16 import load_prompts, SimClassifier, compute_embeddings, ClipWrapper
import torch


def q16_binary_classifier(device):
    prompt_path = 'src/tasks/utils/metrics/q16/prompts.p'
    trained_prompts = load_prompts(prompt_path, device=device)
    clip_model = ClipWrapper(device)
    classifier = SimClassifier(trained_prompts, device)
    classifier.to(device)
    return clip_model, classifier

def harm_eval(clip_model, classifier, img_path, device):
    x = compute_embeddings(clip_model, img_path, device)
    y = classifier(x)
    label = torch.argmax(y, dim=0) # 0 for benign, 1 for malicious
    return label.item(), torch.softmax(y, dim=-1).squeeze()
    


# from transformers import AutoImageProcessor, ResNetForImageClassification
# import torch

# def imagenet_ResNet50(device):
#     processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#     model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
#     model.to(device)
#     return processor, model

# def object_eval(classifier, img, processor, device):
#     with torch.no_grad():
#         inputs = processor(img, return_tensors="pt")
#         inputs.to(device)
#         logits = classifier(**inputs).logits

#     # model predicts one of the 1000 ImageNet classes
#     predicted_label = logits.argmax(-1).item()
#     # print(predicted_label)
#     # print(classifier.config.id2label[predicted_label])
#     return predicted_label, torch.softmax(logits, dim=-1).squeeze()