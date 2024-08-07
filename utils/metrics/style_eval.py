from transformers import pipeline
def init_classifier(device,path):
    return pipeline('image-classification',model=path,device=device)

def style_eval(classifier,img):
    return classifier(img,top_k=129)