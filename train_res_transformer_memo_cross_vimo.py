import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer_memo_cross import ResidualTransformer
from models.mask_transformer.transformer_memo_trainer import ResidualTransformerTrainer
from models.vq.model import RVQVAE
from models.tc_clip.tc_clip import TCCLIP_VE
from custom_clip import clip

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain

from data.vimo_dataset import VimoDataset, train_pipeline, val_pipeline
from models.t2m_eval_wrapper import EvaluatorModelWrapper

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.load_state_dict(ckpt[model_key])
    return vq_model, vq_opt

def load_clip_to_cpu(model_arch):
    print(f'Loading CLIP Model with {model_arch}')
    backbone_name = model_arch
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, './checkpoints/clip')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"vision_model": "TCVisionTransformer",
                    "vision_block": "TCAttentionBlock",
                    "text_block": "ResidualAttentionBlock",
                    "use_custom_attention": True,
                    "context_length": 30,
                    "temporal_length": 16,
                    "vision_depth": 0,
                    "language_depth": 1,
                    "vision_ctx": 0,
                    "language_ctx": 0,
                    # TC-CLIP
                    "positional_embedding_type": "space",
                    "local_global_bias": True,
                    "context_token_k": 96,
                    "seed_token_a": 0.3,
                    "tome_r": 100,
                    "tome_d": 0
                    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_tcclip_model(model_path, model):
    print(f"Loading tcclip model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    load_state_dict = checkpoint['model']

    for param_name in list(load_state_dict.keys()):
        if 'image_encoder' not in param_name:
            del load_state_dict[param_name]
    
    msg = model.load_state_dict(load_state_dict, strict=False)
    print(f"loaded model: {msg}")
    del checkpoint
    torch.cuda.empty_cache()
    return model

def prepare_video_encoder(clip_version):
    clip_model = load_clip_to_cpu(clip_version)
    video_encoder = TCCLIP_VE(clip_model)

    for param in video_encoder.parameters():
        param.requires_grad_(False)
    
    video_encoder.float()
    del clip_model
    torch.cuda.empty_cache()
    video_encoder.to(opt.device)

    video_encoder = load_tcclip_model('./checkpoints/tcclip/zero_shot_k400_tc_clip_newkeys.pth', video_encoder)
    return video_encoder

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/res/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    
    if opt.dataset_name == "vimo":
        opt.data_root = '../Data/VIMO/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.joints_num = 22
        # opt.max_motion_len = 55
        opt.max_motion_length = 200
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/vimo/Comp_v6_KLD005/opt.txt'
        clip_version = 'ViT-B/16'
    else:
        raise KeyError('Dataset Does Not Exist')

    vq_model, vq_opt = load_vq_model()
    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    video_encoder = prepare_video_encoder(clip_version)

    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='video',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                          # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                          share_weight=opt.share_weight,
                                          clip_version=clip_version,
                                          opt=opt)

    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())

    # print(res_transformer)
    # print("Total parameters of res_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params += pc_transformer

    print('Total parameters of res_transformer model: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_dataset = VimoDataset(opt, mean, std, data_prefix=opt.data_root, ann_file=opt.train_txt, pipeline=train_pipeline)
    val_dataset = VimoDataset(opt, mean, std, data_prefix=opt.data_root, ann_file=opt.test_txt, pipeline=val_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    print('Preprocessing training data...')
    train_loader = [batch_data for batch_data in tqdm(train_loader)]
    print('Preprocessing validation data...')
    val_loader = [batch_data for batch_data in tqdm(val_loader)]
    eval_val_loader = val_loader

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model, video_encoder)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)
    