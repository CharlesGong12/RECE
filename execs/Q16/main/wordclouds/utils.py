import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import copy
import multidict as multidict
import scipy.io
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import PIL
import glob
from numpy import genfromtxt
import multiprocessing
import itertools
from tqdm.contrib.concurrent import process_map

wc_max_font_size = 40
wc_width, wc_height = 400, 300


def compute_chisquared(observed, expected, use_not_expected=False):
    expected_elems = [item[0] for item in expected.items()]
    expected_value_min = np.min([item[1] for item in expected.items()])
    weighted_freq = dict()
    sum_chisquared = 0
    cnt = 0
    for item in tqdm(observed.items()):
        freq_diff = 0
        if item[0] in expected_elems:
            expected_value = expected[item[0]]
        else:
            if use_not_expected:
                expected_value = expected_value_min
            else:
                expected_value = 100.0

        if item[1] >= expected_value:
            freq_diff = ((item[1] - expected_value) ** 2) / (expected_value)
            sum_chisquared += freq_diff
            cnt += 1

        weighted_freq[item[0]] = freq_diff
    return weighted_freq, sum_chisquared, cnt


def _compute_chisquared_item(item, expected, expected_elems, weighted_freq):
    # print('Test', item)
    freq_diff = 0
    if item[0] in expected_elems:
        expected_value = expected[item[0]]
    else:
        expected_value = 100.0

    if item[1] >= expected_value:
        freq_diff = ((item[1] - expected_value) ** 2) / (expected_value)

    weighted_freq[item[0]] = freq_diff
    # return 0#freq_diff


def func_star(a_b):
    return _compute_chisquared_item(*a_b)


def compute_chisquared_parallel(observed, expected, use_not_expected=False,
                                max_workers=5,chunksize=100000):
    expected_elems = [item[0] for item in expected.items()]
    # expected_value_min = np.min([item[1] for item in expected.items()])
    # pool = multiprocessing.Pool(4)

    # res = process_map(_compute_chisquared_item, observed.items(), max_workers=max_workers, chunksize=1000)
    with multiprocessing.Manager() as manager:
        weighted_freq = manager.dict()
        with manager.Pool(processes=max_workers) as p:
            with tqdm(total=len(observed.items())) as pbar:
                for _ in p.imap_unordered(func_star, zip(observed.items(),
                                                         itertools.repeat(expected),
                                                         itertools.repeat(expected_elems),
                                                         itertools.repeat(weighted_freq)),
                                          chunksize=chunksize):
                    pbar.update()
        # pool.map(calc_stuff, range(0, 10 * offset, offset))
        weighted_freq_res = dict(weighted_freq)
        # print(weighted_freq_res.values()[:10])
    return weighted_freq_res, None, None  # weighted_freq_res, None, None


def weighted_wc(caption_text_inappr, caption_text_other, save_path, use_bigrams=True, collocation_threshold=30,
                verbose=False):
    wc_noninapp = wc_text(caption_text_other, 10000000, save_path=None,
                          collocations=use_bigrams, collocation_threshold=collocation_threshold)
    wc_inapp = wc_text(caption_text_inappr, 10000000, save_path=None,
                       collocations=use_bigrams, collocation_threshold=collocation_threshold)

    print('done calc. word freq.')
    print([(item[0], item[1]) for item in wc_inapp.words_.items()][:20])
    print('----')
    print([(item[0], item[1]) for item in wc_noninapp.words_.items()][:20])

    relative_scaling = 0.25  # 'auto'
    use_not_expected = False

    print('compute chi-squared weights')
    weighted_freq, sum_chisquared, cnt_chisquared = compute_chisquared(wc_inapp.words_,
                                                                       wc_noninapp.words_,
                                                                       use_not_expected=use_not_expected)
    file_name = os.path.join(save_path,
                             f"wc_weighted_chi-squared_scaling{str(relative_scaling).replace('.', '-')}.png")

    print('compute final wordcloud')
    wc = wc_freq(weighted_freq, max_words=500, relative_scaling=relative_scaling,
                 save_path=None if verbose else file_name,
                 show=verbose)

    print('Chisquared freq', sum_chisquared / cnt_chisquared)
    print(list(wc.words_.items())[:20])


def getFrequencyDictForText(sentence, regex_words, word_boundary=False):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    regex_words_ = copy.deepcopy(regex_words)
    if word_boundary:
        regex_words_ = '\\b' + regex_words_.replace('|', '\\b|\\b') + '\\b'
    for text in sentence.split(","):
        if re.match(regex_words_, text):
            val = tmpDict.get(text.lower(), 0)
            tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict


def getFrequencyDictForImgTextPairs(img_text_dict, regex_words, word_boundary=False):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}
    img_cnt_dict = {}
    # making dict for counting frequencies
    regex_words_ = copy.deepcopy(regex_words)
    if word_boundary:
        regex_words_ = '\\b' + regex_words_.replace('|', '\\b|\\b') + '\\b'
    for img_path in tqdm(list(img_text_dict.keys())):

        sentences = img_text_dict[img_path]
        for sentence in sentences:
            found = False
            for text in sentence.split(' '):
                if re.match(regex_words_, text):
                    val = tmpDict.get(text.lower(), 0)
                    val_img_cnt = img_cnt_dict.get(img_path, {'cnt': 0, 'words': [], 'sentences': []})
                    tmpDict[text.lower()] = val + 1
                    val_img_cnt['cnt'] += 1
                    val_img_cnt['words'] += [text.lower()]
                    img_cnt_dict[img_path] = val_img_cnt
                    found = True
            if found:
                img_cnt_dict[img_path]['sentences'] = sentences
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict, img_cnt_dict


def makeImage(text, relative_scaling=0.1):
    wc = WordCloud(max_font_size=wc_max_font_size, background_color="white", max_words=1000, colormap='Dark2',
                   width=wc_width, height=wc_height, relative_scaling=relative_scaling)
    # generate word cloud
    wc.generate_from_frequencies(text)
    return wc


def wc_text(text, max_words, save_path=None, show=False, collocations=False, collocation_threshold=30):
    wc = WordCloud(max_font_size=wc_max_font_size, max_words=max_words, width=wc_width, height=wc_height,
                   relative_scaling='auto', collocations=collocations, collocation_threshold=collocation_threshold,
                   background_color="white", colormap='Dark2').generate(
        text)
    plot_wc(wc, save_path, show)
    return wc


def plot_wc(wc, save_path=None, show=False):
    if save_path is not None:
        img = wc.to_image()
        img.save(save_path, quality=95, optimize=True)
        # wc.to_file(save_path)
    if show:
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()


def wc_freq(freq, max_words, save_path=None, show=False, relative_scaling='auto',
            _wc_width=wc_width, _wc_height=wc_height, res=(1, 1), scale=1):
    wc = WordCloud(max_font_size=wc_max_font_size, max_words=max_words, width=_wc_width, height=_wc_height,
                   relative_scaling=relative_scaling, collocations=False,
                   background_color="white", colormap='Dark2', scale=scale)
    wc.generate_from_frequencies(freq)
    plot_wc(wc, save_path, show)
    return wc


def wc_caption(text_captions, temps, save_path):
    wordcloud_caption_freq = WordCloud(max_font_size=wc_max_font_size, max_words=150, width=wc_width, height=wc_height,
                                       relative_scaling=0.25,
                                       collocations=False, background_color="white", colormap='Dark2').generate(
        text_captions)

    # store to file
    if temps is not None:
        file_name = os.path.join(save_path,
                                 f"wordcloud_caption_freq_temp{str(temps)}.png")
    else:
        file_name = os.path.join(save_path,
                                 "wordcloud_caption_freq.png")

    wordcloud_caption_freq.to_file(file_name)
    fig = plt.figure(figsize=(16, 6))
    words = list(wordcloud_caption_freq.words_.keys())[:40]
    freq = [wordcloud_caption_freq.words_[f] for f in words]
    plt.xticks(rotation=90)
    plt.bar(words, freq)
    plt.savefig(os.path.join(save_path, "barchart_caption.png"))
    # plt.show()
    plt.close()


def wc_bad_words_from_imgpaths(img_text_dict, regex, save_path=None, save_path_suffix='', show=False):
    freq_dict, imgpath_counts = getFrequencyDictForImgTextPairs(img_text_dict,
                                                                regex, word_boundary=True)
    wordcloud_caption_bad_freq = makeImage(freq_dict, relative_scaling=0.3)

    save_path_file = os.path.join(save_path, f"wordcloud_badwords{save_path_suffix}.png")
    plot_wc(wordcloud_caption_bad_freq, save_path_file, show)

    fig = plt.figure(figsize=(16, 6))
    words = list(wordcloud_caption_bad_freq.words_.keys())
    freq = [wordcloud_caption_bad_freq.words_[f] for f in words]
    plt.xticks(rotation=90)
    plt.bar(words, freq)
    save_path_file = os.path.join(save_path, f"histogram_badwords{save_path_suffix}.png")
    if save_path is not None:
        plt.savefig(save_path_file)
    if show:
        plt.show()
    plt.close()
    return imgpath_counts


def wc_classes_freq(class_text, regex_classes, save_path):
    wordcloud_classes_freq = makeImage(getFrequencyDictForText(class_text, regex_classes), relative_scaling=0.25)
    wordcloud_classes_freq.to_file(os.path.join(save_path, "wordcloud_classes_freq.png"))

    fig = plt.figure(figsize=(16, 6))
    words = list(wordcloud_classes_freq.words_.keys())[:100]
    freq = [wordcloud_classes_freq.words_[f] for f in words]
    plt.xticks(rotation=90)
    plt.bar(words, freq)
    plt.savefig(os.path.join(save_path, "barchart_freq.png"))
    # plt.show()
    plt.close()
    return wordcloud_classes_freq


def wc_classes_infreq(wordcloud_classes_freq, class_text, save_path):
    frequencies_classes_infreq = dict()
    cnt = 0

    # print(wordcloud_classes_freq.words_)
    for word in wordcloud_classes_freq.words_:
        # print(wordcloud_classes_freq.words_[word])
        if class_text.find(word) >= 0 and wordcloud_classes_freq.words_[word] < 1:
            frequencies_classes_infreq[word] = 1 - wordcloud_classes_freq.words_[word]
            cnt += 1
    print('#Words', cnt)
    wordcloud_classes_infreq = WordCloud(max_font_size=wc_max_font_size, max_words=150,
                                         width=wc_width, height=wc_height, relative_scaling=0.25,
                                         collocations=False, background_color="white", colormap='Dark2')
    wordcloud_classes_infreq.generate_from_frequencies(frequencies_classes_infreq)
    wordcloud_classes_infreq.to_file(os.path.join(save_path, "wordcloud_classes_infreq.png"))

    fig = plt.figure(figsize=(16, 6))
    words = list(wordcloud_classes_infreq.words_.keys())[:100]
    freq = [wordcloud_classes_infreq.words_[f] for f in words]
    plt.xticks(rotation=90)
    plt.bar(words, freq)
    plt.savefig(os.path.join(save_path, "barchart_infreq.png"))
    # plt.show()
    plt.close()

    return wordcloud_classes_infreq


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_samples(model, inputs_list, temp_=0.9, num_generations=5):
    # embed + generate response
    # embeddings = []
    response = []
    for inputs_ in inputs_list:
        embedding = model.embed(inputs_)
        set_seed(42)
        responses_img = []
        # for _ in range(num_generations):
        out = model.generate(embedding.repeat(num_generations, 1, 1), temperature=temp_)
        response.append(out)
    return response
    # embeddings = torch.stack(embeddings)


def create_captions(images, temps, num_generations, save_path):
    config_path = "/workspace/repositories/multimodal_fewshot/configs/magma_rn50x16-mp1-config.yml"
    ckpt_path = "/workspace/repositories/multimodal_fewshot/models/mp_rank_00_model_states-step30000.pt"
    model_dir = "/workspace/repositories/multimodal_fewshot/models"

    texts_res = {}
    img_text = {}
    prompt = 'A picture of'
    prompts = [prompt]
    model, transforms, tokenizer = None, None, None

    for temp in temps:
        texts_res[temp] = []
        img_text[temp] = {}
        caption_save_path = os.path.join(save_path,
                                         f"caption_generated_temp{temp}_#gen{num_generations}.p")
        caption_img_save_path = os.path.join(save_path,
                                             f"caption_img_generated_temp{temp}_#gen{num_generations}.p")
        if not os.path.isfile(caption_save_path) or not os.path.isfile(caption_img_save_path):
            if model is None:
                model, transforms, tokenizer = get_multimodal_model(config_path,
                                                                    ckpt_path=ckpt_path,
                                                                    model_dir=model_dir)
                model.cuda()
                model.half()
                model.eval()
            print('model loaded')

            for image_path in tqdm(images, disable=os.environ.get("DISABLE_TQDM", False)):
                imgs = [PIL.Image.open(image_path)] * len(prompts)
                inputs_ = [[transforms(img), tokenizer(prompt, return_tensors='pt')['input_ids']] for img, prompt in
                           zip(imgs, prompts)]
                res = generate_samples(model, inputs_, temp, num_generations=num_generations)
                texts_res[temp] += res
                img_text[temp][os.path.basename(image_path)] = res

            if not os.path.isfile(caption_save_path):
                os.makedirs(save_path, exist_ok=True)
                pickle.dump(texts_res[temp], open(caption_save_path, 'wb'))
            if not os.path.isfile(caption_img_save_path):
                os.makedirs(save_path, exist_ok=True)
                pickle.dump(img_text[temp], open(caption_img_save_path, 'wb'))
        else:
            texts_res[temp] = pickle.load(open(caption_save_path, 'rb'))
            img_text[temp] = pickle.load(open(caption_img_save_path, 'rb'))
    print('created captions')
    return texts_res, img_text


def load_offensive_captions(caption_paths, csv_path):
    path_offandnonoffText = os.path.join(caption_paths, "text_inapp_and_noninapp.p")

    if not os.path.isfile(path_offandnonoffText):

        ## load all caption
        img_text = load_captions(caption_paths)
        img_path_offending = readoffendingimages_csv(csv_path, threshold=.5)

        # temps = list(img_text.keys())
        temps = [0.1, 0.4]
        cnt_captions = 0
        tmp_paths = []
        img_caption_dict = {}

        def replace_characters(text):
            new_text = text.replace('<PERSON>', 'PERSON').lower()  # .replace('.', '')
            new_text = new_text.replace(',', '').replace('…', '')
            new_text = new_text.replace('"', '').replace(')', '').replace('(', '')
            # new_text
            return new_text

        for temp in temps:
            for img_path in list(img_text[temp].keys()):
                img_id = img_path.split('.')[0]
                if img_id not in img_caption_dict:
                    img_caption_dict[img_id] = []
                for img_captions in img_text[temp][img_path]:
                    for caption in img_captions:
                        tmp_paths += [img_path]
                        cnt_captions += 1
                        img_caption_dict[img_id] += [replace_characters(caption)]  # + '\n'
                # print(t[0])
        print('#Captions:', cnt_captions)
        print('#Images:', len(np.unique(tmp_paths)))

        ## split captions into offending and non-offending images
        caption_text_offimg = ''
        caption_text_nonoffimg = ''
        for img_id in tqdm(list(img_caption_dict.keys())):
            if img_id in img_path_offending:
                caption_text_offimg += ' ' + ' '.join(img_caption_dict[img_id])
            else:
                caption_text_nonoffimg += ' ' + ' '.join(img_caption_dict[img_id])
        pickle.dump({'off': caption_text_offimg, 'nonoff': caption_text_nonoffimg}, open(path_offandnonoffText, "wb"))
    else:
        data_text = pickle.load(open(path_offandnonoffText, "rb"))
        caption_text_nonoffimg = data_text['nonoff']
        caption_text_offimg = data_text['off']
    return caption_text_nonoffimg, caption_text_offimg


def create_class_text(images, get_class_text, save_path):
    class_text_save_path = os.path.join(save_path, "class_text.p")
    class_text_dict_save_path = os.path.join(save_path, "class_text_dict.p")
    if not (os.path.isfile(class_text_save_path) and os.path.isfile(class_text_dict_save_path)):
        class_text = ''
        class_images = dict()
        for image_path in tqdm(images):
            im_str = image_path
            img_texts = get_class_text(im_str)
            for img_text in img_texts:
                class_text += ',' + img_text
                if img_text not in list(class_images.keys()):
                    class_images[img_text] = {'cnt': 1, 'images': [image_path]}
                else:
                    class_images[img_text]['cnt'] += 1
                    class_images[img_text]['images'] += [image_path]
        pickle.dump(class_text, open(class_text_save_path, 'wb'))
        pickle.dump(class_images, open(class_text_dict_save_path, 'wb'))
    else:
        class_text = pickle.load(open(class_text_save_path, 'rb'))
        class_images = pickle.load(open(class_text_dict_save_path, 'rb'))
    return class_images, class_text


def readoffendingimages_csv(data_path, threshold=0.50):
    data = list(np.loadtxt(open(data_path, "rb"), delimiter=",", skiprows=1, dtype=str))
    data = sorted(data, key=lambda d: -float(d[2]))
    files_ = [e[3].split('.')[0] for e in data if float(e[2]) >= threshold]

    return files_


def find_images(image_paths, types=None):
    if types is None:
        types = ('/*.JPEG', '/*.png', '/*.jpg', '/*/*.JPEG', '/*/*.png', '/*/*.jpg')  # the tuple of file types
    files_grabbed = []
    for files_ in types:
        files_grabbed.extend(glob.glob(image_paths + files_))
    if len(files_grabbed) == 0:
        raise ValueError('no data found')
    return files_grabbed


def imagenet_images(dataset_path, files):
    print('Loading images ...')
    image_paths = find_images(dataset_path, types=None)
    print('Images:', len(image_paths))
    imagenet_path_dict = dict()
    for image_path in image_paths:
        imagenet_path_dict[os.path.basename(image_path).split('.')[0]] = image_path
    images = [imagenet_path_dict[file_] for file_ in files]
    print('Images:', len(images))
    return images


def openimages_images(dataset_path, files):
    images = [os.path.join(dataset_path, file_) + '.jpg' for file_ in files]
    return images


def load_captions(file_path):
    file_template = '*/caption_img_generated_temp*.p'
    files_grabbed = glob.glob(os.path.join(file_path, file_template))
    print(f'{len(files_grabbed)} caption files found')
    img_text = {}
    for file_caption in files_grabbed:
        temp = float(os.path.basename(file_caption).split('temp')[1].split('_')[0])
        if temp not in list(img_text.keys()):
            img_text[temp] = {}
        data = pickle.load(open(file_caption, 'rb'))
        for d_key in data.keys():
            img_text[temp][d_key] = data[d_key]
    return img_text


def load_captions_from_csv(file_path):
    data = genfromtxt(file_path, delimiter='\t', dtype=str, deletechars="~!@#$%^&*()-=+~\|]}[{';: /?.>,<.")
    data = data[:, 1]
    return ' '.join(data)


def class_annotation_imagenet(images, dataset_path="/workspace/datasets/imagenet1k/"):
    imagenet_dir_class = dict()
    mat = scipy.io.loadmat(os.path.join(dataset_path, "meta.mat"))
    tmp = [(e[0][1], e[0][2]) for e in mat["synsets"]]
    for t_ in tmp:
        imagenet_dir_class[t_[0].item()] = t_[1].item().split(',')[0]

    def get_class_text_(img_path):
        imgnet_dirname = os.path.basename(os.path.dirname(img_path))
        text_ = imagenet_dir_class[imgnet_dirname]
        return [text_]

    return get_class_text_


def class_annotations_openimages(images, dataset_path="/workspace/datasets/openimagesv6"):
    images_ = [os.path.basename(i).replace('.jpg', '') for i in images]

    df = pd.read_csv(os.path.join(dataset_path, "train-annotations-human-imagelabels-boxable.csv"))
    df_classnames = pd.read_csv(os.path.join(dataset_path, "/class-descriptions-boxable.csv"),
                                names=['LabelName', 'Label'])

    df_offensive = df[df["ImageID"].isin(images_)]
    print(len(df_offensive))
    df = df_offensive.to_numpy()
    df_classnames = df_classnames.to_numpy()
    image_ids = df[:, 0]
    image_labels = df[:, 2]
    classnames_labelnames = df_classnames[:, 0]
    classnames_labels = df_classnames[:, 1]

    def get_class_text_(img_path):
        image_id = os.path.basename(img_path).replace('.jpg', '')
        classes = []
        for label_name in image_labels[image_ids == image_id].tolist():
            classes.append(classnames_labels[classnames_labelnames == label_name].item())
        return classes

    return get_class_text_
