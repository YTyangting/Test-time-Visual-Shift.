import os.path as osp
from collections import OrderedDict
import math
import copy
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import trainers.augmix_ops as augmentations
from copy import deepcopy
import torch.backends.cudnn as cudnn
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import math
import json
import random
from torch.utils.data import Dataset
from torch.optim import Optimizer
import numpy as np
from dassl.utils.tpt_tools import Summary, ProgressMeter, accuracy, load_model_weight, set_random_seed
from dassl.utils.tpt_tools import AverageMeter as AverageMeter_TPT
import time
from tqdm import tqdm

from pdb import set_trace as stx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from clip import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.imagenet_templates import IMAGENET_TEMPLATES
from trainers.imagenet_variants import imagenet_classes,imagenet_v_mask
_tokenizer = _Tokenizer()

tpt_to_map = {
    'I': 'imagenet',
    'A': 'imagenet_a',
    'V': 'imagenetv2',
    'R': 'imagenet_r',
    'K': 'imagenet_sketch',
    'flower102': 'flower102',
    'food101': 'food101',
    'dtd': 'dtd',
    'aircraft': 'aircraft',
    'ucf101': 'ucf101',
    'eurosat': 'eurosat',
    'caltech101': 'caltech101',
    'cars': 'stanford_cars',
    'pets': 'oxford_pets',
    'sun397': 'sun397'
}
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                    "vision_depth": 0,
                    "language_depth": 0, 
                    "vision_ctx": 0,
                    "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    #model = clip.build_model(state_dict or model.state_dict())
    return model

class VisionEncoder(nn.Module):
    def __init__(self, clip_model): #, image_weight
        super().__init__()
        self.visual = clip_model.visual  # CLIP's visual encoder
        self.ln_pre = self.visual.ln_pre
        self.transformer = self.visual.transformer
        self.ln_post = self.visual.ln_post
        self.proj = self.visual.proj
        self.dtype = clip_model.dtype
        self.conv1 = self.visual.conv1
        self.class_embedding = self.visual.class_embedding
        self.positional_embedding = self.visual.positional_embedding
    def forward(self, x, ctx_v):
        ctx_v = ctx_v.expand(x.shape[0], -1, -1).half()
        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.type(self.dtype)
        x = torch.cat([x, ctx_v], dim=1)
        # x = torch.cat([x, ctx_v[:, 0, :, :]], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x
    
class TextEncoder(nn.Module):
    def __init__(self,  clip_model):
        super().__init__()
        #self.transformer = clip_model.transformer.resblocks
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, ctx_t):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        
        return x
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.TVS.N_CTX_VISION
        print('................................................................')
        print(cfg.DATASET.LR)
        dtype = clip_model.dtype
        
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim=ctx_dim
        self.device = clip_model.visual.conv1.weight.device
        ctx_init=cfg.TRAINER.TVS.CTX_INIT
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        #self.classnames = classnames

        n_pro = cfg.TRAINER.TVS.N_CTX_VISION
        vct_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        v_ctx_vectors = torch.empty(n_pro, vct_dim, dtype=self.dtype)
        nn.init.normal_(v_ctx_vectors, std=0.005)
        #self.v_ctx = nn.Parameter(v_ctx_vectors)
        self.v_ctx = v_ctx_vectors.to(device)
        self.v_ctx_init_state=v_ctx_vectors.detach().clone()

        self.alpha_init_state=torch.tensor(1.0)
        self.alpha = nn.Parameter(nn.Parameter(torch.tensor(1.0)))
        # self.beta_init_state=torch.tensor(1.0)
        # self.beta = nn.Parameter(nn.Parameter(torch.tensor(1.0)))
        #res_cpt=torch.zeros(n_cls ,ctx_dim, dtype=self.dtype).to(device)

        res_ipt=torch.zeros(1,ctx_dim, dtype=self.dtype).to(device)
        
        #scale  and special shift
        # res_ipt=torch.zeros(64,ctx_dim, dtype=self.dtype).to(device)
        # res_ip=torch.ones(64,1, dtype=self.dtype).to(device)
        
        #affine
        # res_ipt = torch.eye(ctx_dim)  
        # res_ip = torch.zeros(ctx_dim)

        #self.res_cpt_init_state = res_cpt.detach().clone()
        self.res_ipt_init_state = res_ipt.detach().clone()
        #self.res_ip_init_state = res_ip.detach().clone()
        #self.text_feature_residuals = nn.Parameter(res_cpt)
        #self.text_feature_residuals = res_cpt
        self.image_feature_residuals = nn.Parameter(res_ipt)
        #self.image_feature_scale = nn.Parameter(res_ip)
    def reset_alpha(self,adaptive_scale):
        alpha=adaptive_scale.squeeze(0).detach().clone()
        self.alpha.copy_(alpha)
    def reset_res_cpt(self):
        #res_cpt=self.res_cpt_init_state 
        res_ipt=self.res_ipt_init_state
        #res_ip=self.res_ip_init_state
        #self.text_feature_residuals.copy_(res_cpt)
        self.image_feature_residuals.copy_(res_ipt)
        #self.image_feature_scale.copy_(res_ip)

        alpha=self.alpha_init_state
        self.alpha.copy_(alpha)
        # beta=self.beta_init_state
        # self.beta.copy_(beta)
    def reset_vctx(self):
        v_ctx_vectors=self.v_ctx_init_state
        self.v_ctx.copy_(v_ctx_vectors) # to be optimized
    def set_prompt_init_states(self):
        '''
        Store the initial prompts
        '''
        
        self.v_ctx_init_state=self.v_ctx.detach().clone()
        #self.t_ctx_init_state=self.t_ctx.detach().clone()
    def tas_ward(self,base_image_features,test):
        #self.register_buffer("base_text_features", base_text_features)
        # self.register_buffer("base_image_features", base_image_features)
        # tr=beta*base_text_features + (1-beta) * self.text_feature_residuals   # t + a * x
        # ir=beta*base_image_features + (1-beta) * self.image_feature_residuals   # t + a * x
        #tr=base_text_features + self.alpha * self.text_feature_residuals   # t + a * x
        ir=base_image_features + self.alpha*self.image_feature_residuals   # t + a * x
        #ir=base_image_features + self.image_feature_residuals   # t + a * x
        
        #scale
        # if test:
        #     #ir=self.image_feature_residuals[0] *base_image_features    # t + a * x
        #     ir=self.image_feature_scale[0]*base_image_features + self.image_feature_residuals[0]   # t + a * x
        # else:
        #     #ir=self.image_feature_residuals *base_image_features    # t + a * x
        #     ir=self.image_feature_scale*base_image_features + self.image_feature_residuals   # t + a * x
        #affine
        #ir=F.linear(base_image_features, self.image_feature_residuals, self.image_feature_scale)
        
        #return tr,ir
        return ir
    def forward(self):
        v_ctx = self.v_ctx
        ctx=0
        prompts=0
        tokenized_prompts=0
        return prompts, ctx,tokenized_prompts,v_ctx
        #return prompts, ctx,self.tokenized_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device,dataset_tpt):
        super().__init__()

        for p in clip_model.parameters():
            p.requires_grad = False
        self.n_cls=len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model,device)
        #self.vision_prompt_learner = VisionPromptLearner(args, clip_model)
        self.image_encoder = clip_model.visual
        #self.image_encoder = VisionEncoder(clip_model)
        #self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        #self.model = clip_model
        self.classname = classnames
        #self.templates = IMAGENET_TEMPLATES
        self.device = device
        self.dataset=dataset_tpt
        #self.text_feature=torch.load("text_feature/"+tpt_to_map[self.dataset]+"_text_feature_gpt4_template.pth",map_location=torch.device(self.device))
        self.text_feature=torch.load("text_feature/"+tpt_to_map[self.dataset]+"_text_feature_gpt4_x_template.pth",map_location=torch.device(self.device))
        self.temperature=0.5
        self.class_embeds =torch.load("text_feature/"+tpt_to_map[self.dataset]+"_text_feature_clip.pth",map_location=torch.device(self.device))
        self.prompt_embeds=torch.load("confidence_text_feature/"+tpt_to_map[self.dataset]+"_text_feature_template.pth",map_location=torch.device(self.device))
        self.concept_embeds=torch.load("text_feature/"+tpt_to_map[self.dataset]+"_text_feature_gpt4_template.pth",map_location=torch.device(self.device))
        self.concept_embeds1=torch.load("text_feature/"+tpt_to_map[self.dataset]+"_text_feature_gpt4_x_template.pth",map_location=torch.device(self.device))
        self.describe_embeds=torch.load("text_feature/"+tpt_to_map[self.dataset]+"_text_feature_gpt4.pth",map_location=torch.device(self.device))
        self.similarty=torch.load("similarty/"+tpt_to_map[self.cfg.DATASET.TPT]+".pth",map_location=torch.device(self.device))
    def forward(self, image,test=False):
        logit_scale = self.logit_scale.exp()
        #prompts, ctx_t,tokenized_prompts,ctx_v= self.prompt_learner()
        text_features = self.text_feature
        image_features_zs = self.image_encoder(image.type(self.dtype))

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if test==False:
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            logits_zs=logit_scale*(image_features_zs@ text_features.t())
            logits_zs=F.softmax(logits_zs,dim=1)
            

            entropy = -torch.sum(logits_zs[:1] * torch.log(logits_zs[:1] + 1e-8), dim=1, keepdim=True)

            entropy_scale = torch.sigmoid((entropy - 2.0) * 2.0) 
            margin_values = 0.9 + 0.2 * entropy_scale[0]
            with torch.no_grad():
                self.prompt_learner.reset_alpha(margin_values)
        image_features=self.prompt_learner.tas_ward(image_features_zs,test)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits=logit_scale*(image_features @ text_features.t())

        if test==True:
            logits=self.feature_combine(image_features)
            #logits=F.softmax(logits/1e-9,dim=-1)
        
        return logits,image_features,self.n_cls
    
    def calculate_entropy(self,probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  
        return entropy
    
    def feature_combine(self,embeddings):
        scale = 100.
        #obtain discrete confidence score for each image"""


        logits_prompt = scale * embeddings @ self.prompt_embeds.t()
        probs_prompt = F.softmax(logits_prompt, dim=1)

        logits_describe = scale * embeddings @ self.describe_embeds.t()
        probs_describe = F.softmax(logits_describe, dim=1)

        # logits_concept = scale * embeddings @ self.concept_embeds.t()
        # probs_concept = F.softmax(logits_concept, dim=1)

        logits_concept1 = scale * embeddings @ self.concept_embeds1.t()
        probs_concept1 = F.softmax(logits_concept1, dim=1)

        entropy_concept1 = self.calculate_entropy(probs_concept1)  
        #entropy_concept = self.calculate_entropy(probs_concept)
        entropy_describe = self.calculate_entropy(probs_describe)
        entropy_prompt = self.calculate_entropy(probs_prompt)

        neg_entropies = torch.stack([
        -entropy_concept1, 
        -entropy_describe, 
        -entropy_prompt
        ], dim=0)  # shape: [batch_size, 5]
        weights = F.softmax(neg_entropies, dim=1)  

        weight_concept1 = weights[0,:]   
        #weight_concept = weights[1,:]
        weight_describe = weights[1,:]
        weight_prompt = weights[2,:]
        #weight_class = weights[4,:]
        #probs_concept * weight_concept.unsqueeze(1) +
        combined_embeds = (
            probs_concept1 * weight_concept1.unsqueeze(1) + 
            probs_prompt * weight_prompt.unsqueeze(1) +
            probs_describe * weight_describe.unsqueeze(1)  
        )

        return combined_embeds
    def reset(self):
        self.prompt_learner.reset_res_cpt()

    def set_prompt_inits(self):
        print("Re-updating prompt initializations to current prompts.")
        self.prompt_learner.set_prompt_init_states()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

ID_to_DIRNAME={
    'PUG': 'PUG_ImageNet',
    'I': 'imagenet/images',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'imagenet-sketch/images',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []

        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()
    

fewshot_datasets = ['dtd', 'flower102', 'food101', 'cars', 'sun397', 
                    'aircraft', 'pets', 'caltech101', 'ucf101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "split_zhou_OxfordFlowers.json"],
    "food101": ["images", "split_zhou_Food101.json"],
    "dtd": ["images", "split_zhou_DescribableTextures.json"],
    "pets": ["images", "split_zhou_OxfordPets.json"],
    "sun397": ["SUN397", "split_zhou_SUN397.json"],
    "caltech101": ["101_ObjectCategories", "split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "split_zhou_UCF101.json"],
    "cars": ["", "split_zhou_StanfordCars.json"],
    "eurosat": ["2750", "split_zhou_EuroSAT.json"]
}

pug_setting_dir = {
    'CRoll': 'Camera_Roll',
    'CPitch': 'Camera_Pitch',
    'CYaw': 'Camera_Yaw',
    'OPitch': 'Object_Pitch',
    'ORoll': 'Object_Roll',
    'OScale': 'Object_Scale',
    'OTexture': 'Object_Texture',
    'OYaw': 'Object_Yaw',
    'SLight': 'Scene_Light',
    'Worlds': 'Worlds'
}

class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess,augments, n_views=2,
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        #self.aug_list = []
        self.rate = augments
        self.aug_list1 = []
        self.aug_list2 = augmentations.augmentations
        # if augmix is False:
        #     self.aug_list = []
        # else:
        #     self.aug_list = augmentations.augmentations
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        rate=int(self.n_views*self.rate)
        views1 = [augmix(x, self.preprocess, self.aug_list1, self.severity) for _ in range(rate)]
        views2 = [augmix(x, self.preprocess, self.aug_list2, self.severity) for _ in range(self.n_views-rate)]
        return [image] + views1 + views2

@TRAINER_REGISTRY.register()
class TVS(TrainerX):
    def build_pug_dataset(self, set_id, data_root, transform):
        setting = set_id.split('_')[1]
        pug_dir = pug_setting_dir[setting]
        testdir = os.path.join(data_root, ID_to_DIRNAME['PUG'], pug_dir)
        testset = datasets.ImageFolder(testdir, transform=transform)
        return testset

    def build_fewshot_dataset(self, set_id, root, transform, mode='train', n_shot=None):
        if set_id.lower() == 'aircraft':
            return Aircraft(root, mode, n_shot, transform)
        path_suffix, json_path = path_dict[set_id.lower()]
        json_path = os.path.join(root, json_path)
        image_path = os.path.join(root, path_suffix)
        return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)

    def build_dataset(self, set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
        if set_id == 'I':
            # ImageNet validation set
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
            testset = datasets.ImageFolder(testdir, transform=transform)
        elif set_id in ['A', 'K', 'R', 'V']:
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
            testset = datasets.ImageFolder(testdir, transform=transform)
        elif set_id in fewshot_datasets:
            if mode == 'train' and n_shot:
                testset = self.build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
            else:
                testset = self.build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
        elif 'PUG' in set_id:
            testset = self.build_pug_dataset(set_id, data_root, transform=transform)
        else:
            raise NotImplementedError
            
        return testset
    def build_data_loader(self):
        super().build_data_loader()
        
        self.tpt_loader = self.get_tpt_dataloader(self.cfg.TPT)
    def get_tpt_dataloader(self, args):

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
        tpt = args.RUN
        set_id = self.cfg.DATASET.TPT
        augments=self.cfg.DATASET.RATE
        #augments=True
        # if set_id in ['I','A','R','K','V']:
        #     augments=False
        if tpt:
            base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, augments,n_views=args.BATCH_SIZE-1, 
                                            )
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.BATCH_SIZE

        ##这里有疑惑
        
        val_dataset = self.build_dataset(set_id, data_transform, self.cfg.DATASET.ROOT, mode='test')
        # print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=8, pin_memory=True)
        
        return val_loader
    
    def tpt(self):
        """
        Run Test-time prompt Tuning
        """
        self.model.set_prompt_inits()   # Init with current prompts
        
        for name, param in self.model.named_parameters():
            if not self.cfg.TPT.COCOOP: # MaPLe and CoOp
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            else:
                if "text_encoder" not in name:
                    param.requires_grad_(False)

        # define optimizer
        if self.cfg.TPT.COCOOP:
            optimizer = None
            optim_state = None
        else:
            trainable_param = self.model.prompt_learner.parameters()
            #optimizer =ConstrainedAdam(trainable_param, self.cfg.TPT.LR)
            optimizer = torch.optim.AdamW(trainable_param, self.cfg.DATASET.LR)
            optim_state = deepcopy(optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        print('=> Using native Torch AMP. Training in mixed precision.')
        print("number of test samples: {}".format(len(self.tpt_loader.dataset)))

        cudnn.benchmark = True

        results = {}
        set_id = self.cfg.DATASET.TPT
        select=self.cfg.DATASET.SELECT
        results[set_id] = self.test_time_adapt_eval(self.tpt_loader, self.model, optimizer, optim_state, scaler, self.cfg.TPT,select)
        return results
    
    def test_time_adapt_eval(self, val_loader, model, optimizer, optim_state, scaler, args,cfgs):
        batch_time = AverageMeter_TPT('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter_TPT('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter_TPT('Acc@5', ':6.2f', Summary.AVERAGE)
        top10 = AverageMeter_TPT('Acc@10', ':6.2f', Summary.AVERAGE)
        top50 = AverageMeter_TPT('Acc@50', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5, top10, top50],
            prefix='Test: ')
        print("$"*40)
        print(f"Running for {args.BATCH_SIZE} Augmented views")
        print(f"Running for {args.TTA_STEPS} TTA steps")

        # reset model and switch to evaluate mode
        model.eval()
        if not args.COCOOP: # no need to reset cocoop because it's fixed
            with torch.no_grad():
                model.reset()

        all_preds = []
        all_targets= []
        end = time.time()
        a=0
        for i, batch in enumerate(val_loader):
            test=False
            # images, target = self.parse_batch_test(batch)
            images, target = batch
            # assert args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    # images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    images[k] = images[k].to(self.device)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                # images = images.cuda(args.gpu, non_blocking=True)
                images = images.to(self.device)
                image = images
            # target = target.cuda(args.gpu, non_blocking=True)
            target = target.to(self.device)
            if args.RUN:
                images = torch.cat(images, dim=0)

            # reset the tunable prompt to its initial state
            if args.TTA_STEPS > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            #print('.........')
            m=self.test_time_tuning(model, images, optimizer, scaler, args,cfgs)
            if m==target:
                a+=1 
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # if args.COCOOP:
                    #     output = model((image_feature, pgen_ctx))
                    output,_ ,_= model(image,test=True)   
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc10, acc50 = accuracy(output, target, topk=(10,1))

            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            top10.update(acc10[0], image.size(0))
            top50.update(acc50[0], image.size(0))
            if (i+1) % 1024 == 0:
                
                progress.display(i)
        end_time=time.time()
        time_end=end_time - end
        print("end_time", time_end)
        #print(self.updatetime)
        progress.display_summary()

        return [top1.avg, top5.avg]

    def test_time_tuning(self, model, inputs, optimizer, scaler, args,cfgs):
        if args.COCOOP:
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            optimizer = torch.optim.AdamW([pgen_ctx], args.LR)
        selected_idx=None
        n=args.TTA_STEPS
        n = args.TTA_STEPS
        j=0
        while j < n:
            with torch.cuda.amp.autocast():
                if args.COCOOP:
                    output = model((image_feature, pgen_ctx))
                else:
                    #output,_ = model(inputs) 
                    output ,output1,n_cls= model(inputs)  
                    features=output1
                    #output.retain_grad()    

                output_1,selected_idx,orign_entropy= self.select_confident_samples(output, cfgs)
                output1=output1[selected_idx]
                output_dm=self.feature_combine(output1)
                #_, pred =  output_dm.topk(10, 1, True, True)
                #output_dm.retain_grad()  
                
            score=1 
            loss = self.avg_entropy(output_dm,score)      
            #loss = self.avg_entropy(output_prob,score)
            # print(loss) 
            #loss=loss+0.01*F.cross_entropy(output_dm[0], pseudo_label)
            
            j+=1
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # #pred-caculate similarity
            # out_grad=output_dm.grad
            # a=out_grad[selected_idx].half()

            # selected_a = a[:,pred[0]]

            # selected_s = self.similarty[pred[0],:]
            # a=selected_a@selected_s
            # #a=a@self.similarty
            
            # # adam
            # # s = nn.Parameter(output_dm[0].unsqueeze(0))
            # # s.grad = a.sum(dim=0, keepdim=True).to(dtype=s.dtype, device=s.device)
            # # op = torch.optim.Adam([s], lr=args.LR)
            # # op.step()
            # # op.zero_grad()
            # # a=s
            # a= args.LR*a.sum(dim=0, keepdim=True)
            # a = output_dm[0].unsqueeze(0)-a
            # # #2
            # # out_grad=output_dm.grad
            # # a=out_grad[selected_idx].half()@self.similarty
            # # a= args.LR*a.sum(dim=0, keepdim=True)
            # # a = output_dm[0].unsqueeze(0)-a
            
            scaler.step(optimizer)
            scaler.update()
            
        if args.COCOOP:
            return pgen_ctx
        return 1

    
    def calculate_entropy(self,probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1) 
        return entropy
    def feature_combine(self,embeddings):
        
        scale = 100.

        logits_prompt = scale * embeddings @ self.prompt_embeds.t()
        probs_prompt = F.softmax(logits_prompt, dim=1)

        logits_describe = scale * embeddings @ self.describe_embeds.t()
        probs_describe = F.softmax(logits_describe, dim=1)


        logits_concept1 = scale * embeddings @ self.concept_embeds1.t()
        probs_concept1 = F.softmax(logits_concept1, dim=1)

        entropy_concept1 = self.calculate_entropy(probs_concept1)  # 形状: [batch_size]
        #entropy_concept = self.calculate_entropy(probs_concept)
        entropy_describe = self.calculate_entropy(probs_describe)
        entropy_prompt = self.calculate_entropy(probs_prompt)
        #entropy_class = self.calculate_entropy(probs_class)
        
        neg_entropies = torch.stack([
        -entropy_concept1,  
        -entropy_describe, 
        -entropy_prompt
        
        ], dim=0)  
        weights = F.softmax(neg_entropies, dim=1)  
        weight_concept1 = weights[0,:]    
        #weight_concept = weights[1,:]
        weight_describe = weights[1,:]
        weight_prompt = weights[2,:]
        # weight_class = weights[4,:]
        #probs_concept * weight_concept.unsqueeze(1) +
        combined_embeds = (
            probs_concept1 * weight_concept1.unsqueeze(1) +  
            probs_describe * weight_describe.unsqueeze(1) +
            probs_prompt * weight_prompt.unsqueeze(1)
        )

        return combined_embeds
    def discrete_certainty_estimate(self,embeddings,select):
        
        scale = 100.
        #obtain discrete confidence score for each image"""
        class_embeds =self.class_embeds
        prompt_embeds=self.prompt_embeds
        describe_embeds=self.describe_embeds
        concept_embeds1=self.concept_embeds1
        concept_embeds=self.concept_embeds
        prompt_embeds_q=self.prompt_embeds_q
        prompt_embeds_h=self.prompt_embeds_h
        logits_class_embeds = scale* embeddings @ class_embeds.t() # compute the logits
        probs_class_embeds = F.softmax(logits_class_embeds, dim=1)
        preds_class_embeds = torch.argmax(probs_class_embeds, dim=1)

        logits_prompt_embeds = scale* embeddings @ prompt_embeds.t() # compute the logits
        probs_prompt_embeds = F.softmax(logits_prompt_embeds, dim=1)
        preds_prompt_embeds = torch.argmax(probs_prompt_embeds, dim=1)

        logits_prompt_embeds_q =scale* embeddings @ prompt_embeds_q.t() # compute the logits
        probs_prompt_embeds_q = F.softmax(logits_prompt_embeds_q, dim=1)
        preds_prompt_embeds_q = torch.argmax(probs_prompt_embeds_q, dim=1)

        ##emseble before 40
        logits_prompt_embeds_h = scale* embeddings @ prompt_embeds_h.t() # compute the logits
        probs_prompt_embeds_h = F.softmax(logits_prompt_embeds_h, dim=1)
        preds_prompt_embeds_h = torch.argmax(probs_prompt_embeds_h, dim=1)

        logits_describe_embeds = scale* embeddings @ describe_embeds.t() # compute the logits
        probs_describe_embeds = F.softmax(logits_describe_embeds, dim=1)
        preds_describe_embeds = torch.argmax(probs_describe_embeds, dim=1)
        
        logits_concept_embeds = scale* embeddings @ concept_embeds.t() # compute the logits
        probs_concept_embeds = F.softmax(logits_concept_embeds, dim=1)
        preds_concept_embeds = torch.argmax(probs_concept_embeds, dim=1)

        logits_concept_embeds1 = scale* embeddings @ concept_embeds1.t() # compute the logits
        probs_concept_embeds1 = F.softmax(logits_concept_embeds1, dim=1)
        preds_concept_embeds1 = torch.argmax(probs_concept_embeds1, dim=1)
        

        if self.cfg.DATASET.TPT in ['ddkdd']:
            print("gh")
        #     prompts_log = torch.stack((preds_concept_embeds, preds_concept_embeds1,preds_describe_embeds,preds_class_embeds, preds_prompt_embeds,preds_prompt_embeds_q,preds_prompt_embeds_h))
        #     #consistency_log = torch.tensor([int(torch.all(prompts_log[:, i] == prompts_log[:, i][0])) for i in range(prompts_log.shape[-1])])
        #     #consistency_log = torch.tensor([int(torch.sum(prompts_log[:, i] == pred)) for i in range(prompts_log.shape[-1])])
        #     consistency_log = torch.tensor([int(torch.sum(prompts_log[:, i] == prompts_log[:, i][0])) for i in range(prompts_log.shape[-1])])
        #     stable_id = torch.nonzero(consistency_log >5).squeeze()
        # #,preds_prompt_embeds_q,preds_prompt_embeds_h
        else:
            prompts_log = torch.stack((preds_concept_embeds,preds_concept_embeds1,preds_describe_embeds,preds_prompt_embeds))
                #consistency_log = torch.tensor([int(torch.all(prompts_log[:, i] == prompts_log[:, i][0])) for i in range(prompts_log.shape[-1])])
                #consistency_log = torch.tensor([int(torch.sum(prompts_log[:, i] == pred)) for i in range(prompts_log.shape[-1])])
            consistency_log = torch.tensor([int(torch.sum(prompts_log[:, i] == prompts_log[:, i][0])) for i in range(prompts_log.shape[-1])])
            stable_id = torch.argsort(consistency_log, descending=True)[:int(consistency_log.size()[0] * select)]
            #stable_id = torch.nonzero(consistency_log >4).squeeze()
        return stable_id
    def define_consistence(self,logits):
        _, pred = logits.topk(1, 1)  
        pred = pred.reshape(-1)
        counts = torch.bincount(pred)  
        most_common_element = torch.argmax(counts).item() 
        index = torch.nonzero(pred == most_common_element).reshape(-1).to(self.device)  
        #index = torch.nonzero(pred == pred[0]).reshape(-1).to(self.device)  
        return index       
    def extended_entropy_selection_with_rules(self, output, cfgs, features=None, 
                                            min_samples=8):

        output_1, entropy_selected_idx, orign_entropy = self.select_confident_samples(output, cfgs)
        
        additional_indices = self.select_additional_samples(output, features, entropy_selected_idx, 
                                                            min_samples)

        all_selected = torch.unique(torch.cat([entropy_selected_idx, additional_indices]))
        
        return all_selected

    def select_additional_samples(self, output, features, existing_indices, min_samples):
        rule = 'max_prob'
        batch_size = output.shape[0]
        existing_mask = torch.zeros(batch_size, dtype=torch.bool, device=output.device)
        existing_mask[existing_indices] = True
        additional_indices = []
        
        if rule == 'max_prob':
            probs = F.softmax(output, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
                
            high_prob_mask = (max_probs > 0.8) & ~existing_mask
            high_prob_indices = torch.where(high_prob_mask)[0]
                
            if len(high_prob_indices) > 0:
                high_prob_probs = max_probs[high_prob_indices]
                _, top_indices = torch.topk(high_prob_probs, 
                                        min(3, len(high_prob_indices)))
                additional_indices.extend(high_prob_indices[top_indices].tolist())
            
        elif rule == 'margin':
            probs = F.softmax(output, dim=1)
            top2_probs, _ = torch.topk(probs, 2, dim=1)
            margins = top2_probs[:, 0] - top2_probs[:, 1]
            large_margin_mask = (margins > 0.3) & ~existing_mask
            large_margin_indices = torch.where(large_margin_mask)[0]
                
            if len(large_margin_indices) > 0:
                large_margins = margins[large_margin_indices]
                _, top_indices = torch.topk(large_margins, 
                                        min(3, len(large_margin_indices)))
                additional_indices.extend(large_margin_indices[top_indices].tolist())
            
        elif rule == 'feature_consistency' and features is not None:
            additional_indices.extend(
                self.select_by_feature_consistency(output, features, existing_mask)
            )

        additional_indices = torch.tensor(list(set(additional_indices)), 
                                        device=output.device, dtype=torch.long)
        
        max_additional = max(0, min_samples - len(existing_indices))
        if len(additional_indices) > max_additional:
            additional_indices = additional_indices[:max_additional]
        return additional_indices

    def select_by_feature_consistency(self, output, features, existing_mask):

        if len(features) < 2:
            return []
        
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        consistency_scores = []
        for i in range(len(features)):
            if existing_mask[i]: 
                continue
                
            similarities = similarity_matrix[i]
            similarities[i] = -1
            top_similarities, _ = torch.topk(similarities, min(3, len(similarities)-1))
            avg_similarity = top_similarities.mean()
            consistency_scores.append((i, avg_similarity.item()))
        
        consistency_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, score in consistency_scores[:2]] 
        
        return selected
    def select_confident_samples1(self,index, logits, top):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        tk = int(batch_entropy.size(0) * top)
        if index.size(0) >= tk:
            entropy_in_index = batch_entropy[index]
            idx_in_index = torch.argsort(entropy_in_index, descending=False)[:tk]
            new_idx = index[idx_in_index]
        else:
            new_idx = torch.argsort(batch_entropy, descending=False)[:tk]
        
        return logits[new_idx], new_idx
    def select_confident_samples(self, logits, topTPT):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        orign_entropy=batch_entropy[:1]
        idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
        
        #return logits[idxTPT], idxTPT
        return  logits[idxTPT],idxTPT, orign_entropy

    def avg_entropy(self, outputs,x):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
        #avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) 
        # y = int(torch.where(x < 10, 10, x))
        # avg_logits, top5_indices = avg_logits.topk(y)
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if self.cfg.TPT.LOADER:
                # if self.cfg.TPT.RUN:
                #     input, label = torch.cat(batch[0]), torch.cat(batch[1])
                #     input, label = input.to(self.device), label.to(self.device)
                
                input, label = batch["img"].to(self.device), batch["label"].to(self.device)
                #input, label = batch[0].to(self.device), batch[1].to(self.device)
            else:
                input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.save_feature_maps()
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        return list(results.values())[0]
    
    ################# TPT CHANGES END #######################

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TVS.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        if cfg.DATASET.TPT =='V':
            classnames_all = imagenet_classes
            classnames = [classnames_all[i] for i in imagenet_v_mask]
        else:
            classnames = self.dm.dataset.classnames
        self.device=3

        self.prompt_embeds=torch.load("confidence_text_feature/"+tpt_to_map[self.cfg.DATASET.TPT]+"_text_feature_template.pth",map_location=torch.device(self.device))
        # self.prompt_embeds_q=torch.load("confidence_text_feature/"+tpt_to_map[self.cfg.DATASET.TPT]+"_text_feature_template_q.pth",map_location=torch.device(self.device))
        # self.prompt_embeds_h=torch.load("confidence_text_feature/"+tpt_to_map[self.cfg.DATASET.TPT]+"_text_feature_template_h.pth",map_location=torch.device(self.device))
        self.concept_embeds=torch.load("text_feature/"+tpt_to_map[self.cfg.DATASET.TPT]+"_text_feature_gpt4_template.pth",map_location=torch.device(self.device))
        self.concept_embeds1=torch.load("text_feature/"+tpt_to_map[self.cfg.DATASET.TPT]+"_text_feature_gpt4_x_template.pth",map_location=torch.device(self.device))
        self.describe_embeds=torch.load("text_feature/"+tpt_to_map[self.cfg.DATASET.TPT]+"_text_feature_gpt4.pth",map_location=torch.device(self.device))
        self.similarty=torch.load("similarty/"+tpt_to_map[self.cfg.DATASET.TPT]+".pth",map_location=torch.device(self.device))
        self.updatetime=0
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.TVS.PREC == "fp32" or cfg.TRAINER.TVS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            
        self.model = CustomCLIP(cfg, classnames, clip_model,self.device,cfg.DATASET.TPT)
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            
        self.model.to(self.device)
        #self.model.image_encoder.to(self.device)
        #self.model.prompt_learner.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.TVS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
        # device_ids = [6,7]
        # self.model= nn.DataParallel(self.model, device_ids=device_ids)
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.TVS.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
        
