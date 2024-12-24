from transformers import AutoModel
# video data
import numpy as np
import os
import cv2
import torch
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube
def retrieve_text(frames, 
                  texts, 
                  models={'viclip':None, 
                          'tokenizer':None},
                  topk=5, 
                  device=torch.device('cuda')):
    # clip, tokenizer = get_clip(name, model_cfg['size'], model_cfg['pretrained'], model_cfg['reload'])
    assert(type(models)==dict and models['viclip'] is not None and models['tokenizer'] is not None)
    clip, tokenizer = models['viclip'], models['tokenizer']
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]

model=AutoModel.from_pretrained("/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/ViCLIP_B_16",trust_remote_code=True)
tokenizer = model.tokenizer
model_tokenizer={"viclip":model,"tokenizer":tokenizer}
print("done")

video = cv2.VideoCapture('example1.mp4')
frames = [x for x in _frame_from_video(video)]

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


# retrieval
text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                   "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
                   "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                   "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
                   "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                   "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                   "A playful dog slides down a snowy hill, wagging its tail with delight.",
                   "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
                   "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                   "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]
texts, probs = retrieve_text(frames, text_candidates, models=model_tokenizer, topk=5)

for t, p in zip(texts, probs):
    print(f'text: {t} ~ prob: {p:.4f}')