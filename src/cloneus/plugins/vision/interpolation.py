import numpy as np
import torch

import cv2
from PIL import Image



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.inference_mode()
def slerp_once(v0:torch.Tensor, v1:torch.Tensor, t:float, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    dot = torch.sum(v0 * v1 / (torch.linalg.norm(v0) * torch.linalg.norm(v1)))
    if torch.abs(dot) > DOT_THRESHOLD or torch.isnan(dot):
        v2 = torch.lerp(v0, v1, t)
        #v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2

def np_lerp(v0:np.ndarray, v1:np.ndarray, t:float):
    return (1 - t) * v0 + t * v1

@torch.inference_mode()
def slerp_range(v0:torch.Tensor, v1:torch.Tensor, num:int, t0=0, t1=1, t_range=None):
    return_tensor = False
    if isinstance(v0, torch.Tensor):
        v0 = v0.float().numpy(force=True)
        return_tensor=True
    if isinstance(v1, torch.Tensor):
        v1 = v1.float().numpy(force=True)
        return_tensor=True

    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return v2

    if t_range is None:
        t_range = np.linspace(t0, t1, num)

    v3 = np.array([interpolation(t, v0, v1) for t in t_range])
    if return_tensor:
        v3 = torch.from_numpy(v3)

    return v3

def pil_to_np(images):
    if isinstance(images, Image.Image):
        images = [images]
    return np.stack([np.array(img).astype(float)/255. for img in images])


def image_lerp(img_frames:list[Image.Image], total_frames=32, t0=0, t1=1, t_range=None, loop_back:bool = False, use_slerp:bool=False):
    # TODO: https://github.com/dajes/frame-interpolation-pytorch/tree/main
    # this could simplify things: https://docs.python.org/3/library/bisect.html#bisect.bisect_left
    # https://github.com/dajes/frame-interpolation-pytorch/blob/main/inference.py
    np_imgs = pil_to_np(img_frames)
    img_pairs = list(zip(np_imgs, np_imgs[1:]))
    if loop_back:
        img_pairs += [(np_imgs[-1], np_imgs[0])]
    
    n_pairs = len(img_pairs)
    frames_per_pair = max(round(total_frames/n_pairs), 3) # need at least 3 for start, interim, end
    surplus = max(0, total_frames - (frames_per_pair*n_pairs))

    trans_images = []
    
    for i,(s,e) in enumerate(img_pairs):
        nf = frames_per_pair
        if i == n_pairs-1:
            nf += surplus # tack extra frames on to last transition
        if use_slerp:
            timages = slerp_range(s, e, num=nf, t0=t0, t1=t1, t_range=t_range).clip(0,1)*255
        else:
            timages = np.array([np_lerp(s,e, t) for t in np.linspace(t0, t1, num=nf)]).clip(0,1)*255
        trans_images.append(timages.astype(np.uint8))
    
    return [Image.fromarray(img) for img in np.concatenate(trans_images, 0)]

def diff_frames(f0, f1, px_thresh=0.02):
    gf0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY).astype(float)
    gf1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY).astype(float)

    return np.abs(gf0-gf1) >= px_thresh*255.0
    
def motion_mask(img_frames:list[Image.Image]|np.ndarray, px_thresh=0.02, qtile=90):
    #rgb_weights = [0.299, 0.587, 0.114] # https://docs.opencv.org/4.10.0/de/d25/imgproc_color_conversions.html
    # gray_resized_gif_float = resized_gif.astype(float).dot(rgb_weights).round()
    framearr = np.array(img_frames)
    h,w = framearr.shape[1],framearr.shape[2]
    frame_diffs = np.array([diff_frames(f0, f1, px_thresh).astype(float) for f0,f1 in zip(framearr, framearr[1:])])
    frame_ptile = np.percentile(frame_diffs, qtile, axis=0)
    return frame_ptile


@torch.inference_mode()
def time_lerp(latents, t=0.50, use_slerp:bool= False, keep_dims:bool=True):
    if use_slerp:
        tmerge_latents = slerp_once(latents[:-1], latents[1:], t)
    else:
        tmerge_latents = torch.lerp(latents[:-1], latents[1:], t)
    if keep_dims:
        return torch.cat([tmerge_latents, latents[[-1]]]) # append last latent frame no blend to keep frame count
    return tmerge_latents

@torch.inference_mode()
def blend_latents(rawimg_latents, outimg_latents, latmo_mask, time_blend=True, keep_dims=True):
    # first, slerp using interpolated motion latents (effectively soft masking)
    latent_blend = slerp_once(rawimg_latents, outimg_latents, t=latmo_mask)
    if time_blend:
        # then, lerp tn,tn+1 forward to smooth out frame transitions
        latent_blend = time_lerp(latent_blend, 0.5, use_slerp=False, keep_dims=keep_dims)

    return latent_blend