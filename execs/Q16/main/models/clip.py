import torch
import clip

class ClipVisionModel(torch.nn.Module):
    def __init__(self, language_model, num_classes, device, fine_tune=False):
        super(ClipVisionModel, self).__init__()
        self.MMM, self.preprocess = clip.load(language_model.split('_')[1], device, jit=False)
        self.MMM.to(device)
        self.MMM.eval()
        if not fine_tune:
            for param in self.MMM.parameters():
                param.requires_grad = False
        self.fc = torch.nn.Linear(1024, num_classes).half()

    def forward(self, x):
        out = self.MMM.encode_image(x)
        out = self.fc(out)
        return out


class ClipSimModel_Infer(torch.nn.Module):
    def __init__(self, args, prompts=None):
        super(ClipSimModel_Infer, self).__init__()
        self.MMM, self.preprocess = clip.load(args.language_model, f'cuda:{args.gpu[0]}', jit=False)
        self.MMM.to(f'cuda:{args.gpu[0]}')
        self.MMM.eval()

        labels_clip_prompt = ['positive', 'negative']
        # labels = ['unpleasant', 'pleasant']
        # labels = ['blameworthy', 'praiseworthy']
        text = clip.tokenize([f"This image is about something {labels_clip_prompt[0]}",
                              f"This image is about something {labels_clip_prompt[1]}"
                              ]).to(f'cuda:{args.gpu[0]}')
        if prompts is not None:
            self.text_features = torch.HalfTensor(prompts).to(f'cuda:{args.gpu[0]}')
            print('Using tuned prompts', self.text_features.shape)
        else:
            self.text_features = self.MMM.encode_text(text)

    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


class ClipSimModel(torch.nn.Module):
    def __init__(self, language_model, gpu, num_classes=2, pos_label=0, explain=False):
        super(ClipSimModel, self).__init__()

        self.MMM, self.preprocess = clip.load(language_model.split('_')[1], f'cuda:{gpu}', jit=False)
        self.MMM.to(f'cuda:{gpu}')
        self.MMM.eval()

        self.labels = ['positive', 'negative']
        if pos_label == 1:
            self.labels = ['negative', 'positive']

        if num_classes == 3:
            self.labels.append('neutral')
        self.num_classes = num_classes

        prompts = [f"This image is about something {label}" for label in self.labels]
        # labels = ['unpleasant', 'pleasant']
        # labels = ['blameworthy', 'praiseworthy']
        text = clip.tokenize(prompts).to(f'cuda:{gpu}')
        #with torch.no_grad():
        text_features = self.MMM.encode_text(text)

        self.prompts = torch.nn.parameter.Parameter(text_features)

    def encode(self, x):
        image_features = self.MMM.encode_image(x)
        return image_features

    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        # text_features = self.MMM.encode_text(self.text)
        text_features_norm = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def initialize_model_clip(num_classes, language_model, device, fine_tune=False):
    input_size = 224
    model_ft = ClipVisionModel(language_model, num_classes, device, fine_tune=fine_tune)
    return model_ft, input_size


def load_finetuned_model_clip(num_classes, device, path):
    input_size = 224
    model_ft = ClipVisionModel('Clip_RN50', num_classes, device)
    model_ft.fc.load_state_dict(torch.load(path))
    return model_ft, input_size