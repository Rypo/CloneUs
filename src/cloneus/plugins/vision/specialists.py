import io
import gc
import re
import time
import glob
import random
import typing
import string
from pathlib import Path
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers import FluxPipeline, FluxTransformer2DModel, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
import diffusers.utils as diffutils 

from transformers import  AutoProcessor, AutoModelForCausalLM, GenerationConfig

from optimum import quanto
import numpy as np
import cv2

import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont 

from spandrel import ImageModelDescriptor, ModelLoader

# from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict, get_peft_model
import peft

from sam2.sam2_image_predictor import SAM2ImagePredictor

from cloneus.plugins.vision import fluximg2img,fluxinpaint, loraops, quantops
from cloneus import cpaths


def fig_to_np(fig:mpf.Figure):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    return img

class Upsampler:#(ImageFormatter):
    def __init__(self, model_name:typing.Literal["4xNMKD-Superscale.pth", "4xUltrasharp-V10.pth","4xRealWebPhoto_v4_dat2.safetensors"], input_mode:typing.Literal['BGR','RGB'], dtype=torch.bfloat16, device='cuda:0'):
        # https://civitai.com/articles/904/settings-recommendations-for-novices-a-guide-for-understanding-the-settings-of-txt2img
        # https://github.com/joeyballentine/ESRGAN-Bot/blob/master/testbot.py#L73

        # https://huggingface.co/uwg/upscaler/blob/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth
        # Image tiling:
        # https://github.com/chaiNNer-org/chaiNNer/tree/c852759c615c151246176ccd71648f9db9b7386f/backend/src/nodes/impl/upscale
        self.input_mode = input_mode
        self.dtype = dtype
        self.device = torch.device(device)
        # NOTE: for some models (e.g. 4xNMKD-Superscale) rgb/bgr does seem to make a difference
        # others are barely perceptible
        # From limited testing: 4xNMKD: RGB ⋙ BGR, 4xRealWeb: RGB ≥ BGR, 4xUltra: BGR ≥ RGB, 
        ext_modeldir = cpaths.ROOT_DIR/'extras/models'
        self.model = self._load_model(ext_modeldir.joinpath(model_name))
    
    @torch.inference_mode()
    def _load_model(self, model_path):
        # load a model from disk
        model = ModelLoader('cuda').load_from_file(model_path)
        # make sure it's an image to image model
        assert isinstance(model, ImageModelDescriptor)
        
        model = model.eval().to(self.device, dtype=self.dtype)
        
        for param in model.model.parameters():
            param.requires_grad_(False)
        
        return model

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        self.model.to(device=device, dtype=dtype)
        return self

    def nppil_to_torch(self, images: np.ndarray|Image.Image|list[Image.Image]) -> torch.FloatTensor:
        if not isinstance(images, np.ndarray):
            if not isinstance(images, list):
                images = [images]
            
            images = np.stack([np.array(img) for img in images]) # .convert("RGB")
            
        if images.ndim == 3:
            images = images[None]
        if self.input_mode == 'BGR':
            images = images[:, :, :, ::-1]  # flip RGB to BGR
        images = np.transpose(images, (0, 3, 1, 2))  # BHWC to BCHW
        images = np.ascontiguousarray(images, dtype=np.float32) / 255.  # Rescale to [0, 1]
        return torch.from_numpy(images)

    def torch_to_pil(self, tensor: torch.Tensor) -> list[Image.Image]:
        arr = tensor.float().cpu().clamp_(0, 1).numpy()
        arr = (arr * 255).round().astype("uint8")
        arr = arr.transpose(0, 2, 3, 1) # BCHW to BHWC
        if self.input_mode == 'BGR':
            arr = arr[:, :, :, ::-1] # BGR -> RGB

        return [Image.fromarray(a, "RGB") for a in arr]
        
    @torch.inference_mode()
    def process(self, img: torch.FloatTensor) -> torch.Tensor:
        img = img.to(self.device, dtype=self.dtype)
        #with torch.autocast(self.model.device.type, self.model.dtype):
        output = self.model(img)#.detach_()
        
        return output
    
        
    @torch.inference_mode()
    def upsample(self, images: np.ndarray|Image.Image|list[Image.Image]) -> list[Image.Image]:
        output = self.process(self.nppil_to_torch(images))
        return self.torch_to_pil(output)
    
    @torch.inference_mode()
    def upscale(self, images: np.ndarray|Image.Image|list[Image.Image], scale:float=None) -> list[Image.Image]:
        images = self.nppil_to_torch(images)
        if scale is None:
            scale = self.model.scale
        b,c,h,w = images.shape
        dest_w = int((w * scale) // 8 * 8)
        dest_h = int((h * scale) // 8 * 8)

        for _ in range(3):
            b,c,h,w = images.shape
            if w >= dest_w and h >= dest_h:
                break
            
            images = self.process(images)

        images = self.torch_to_pil(images)
        if images[0].width != dest_w or images[0].height != dest_h:
            images = [img.resize((dest_w, dest_h), resample=Image.Resampling.LANCZOS) for img in images]

        return images

class SAMPlot:
    def __init__(self, image:Image.Image) -> None:
        self.image = image
    

    def show(self, point_coords=None, box_coords=None, masks=None, point_labels=None):
        fig,ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image)
        
        if point_coords is not None:
            point_coords = np.array(point_coords, ndmin=2)
            if point_labels is None:
                point_labels = np.ones(point_coords.shape[0])
            point_labels = np.array(point_labels)
            self._add_points(point_coords, point_labels, ax)
        
        if box_coords is not None:
            box_coords = np.array(box_coords, ndmin=2)
            for box in box_coords:
                self._add_box(box, ax)
        
        if masks is not None:
            masks = np.array(masks, ndmin=3)
            for mask in masks:
                self._add_mask(mask, ax)
        
        fig.tight_layout()
        return ax

    def show_masks(self, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        fig,axes = plt.subplots(1, len(scores), figsize=(10, 10))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            ax = axes.flat[i]
            ax.imshow(self.image)
            self._add_mask(mask, ax, borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self._add_points(point_coords, input_labels, ax)
            if box_coords is not None:
                # boxes
                self._add_box(box_coords, ax)
            if len(scores) > 1:
                ax.set_title(f"Mask {i+1}, Score: {score[i]:.3f}", fontsize=18)
            ax.axis('off')
        fig.show()
        return axes


    def _add_mask(self, mask, ax, random_color=False, borders = True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        
        ax.imshow(mask_image)

    def _add_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def _add_box(self, box, ax, *, edgecolor='green'):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))    





class SAM2:
    def __init__(self, torch_dtype = torch.bfloat16, device="cuda:0", offload:bool=True, image:Image.Image = None) -> None:
        # https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb
        self.torch_dtype = torch_dtype
        self.device = device
        self.offload = offload
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device, dtype=torch_dtype)
        self.predictor.model.eval()
        #self.predictor.model.to(device, torch_dtype)
        self.active_image = image
    
    def to(self, *args, **kwargs):
        _= self.predictor.model.to(*args, **kwargs)
        return self
    
    def _update_image(self, image:Image.Image):
        if image is not None and self.active_image != image:
            self.active_image = image
            self.predictor.set_image(image)

    def _sorted_output(self, masks:np.ndarray, scores:np.ndarray, logits:np.ndarray):
        sorted_ind = np.argsort(scores)[::-1]
        return masks[sorted_ind],scores[sorted_ind],logits[sorted_ind]

    @torch.inference_mode()
    def predict_points(self, point_coords:np.ndarray, point_label:np.ndarray, image:Image.Image=None, multimask_output=True,):
        '''predict mask,score,logits given (xy)points and pos/neg labels
        
        with one point::
            input_point = np.array([[500, 375]])

            input_label = np.array([1]) -- 1=forground, 0=background

        with two points::
            input_point = np.array([[500, 375], [1125, 625]])
            
            input_label = np.array([1, 1])
        '''
        self._update_image(image)

        with torch.autocast(self.device, self.torch_dtype):
            masks, scores, logits = self.predictor.predict(np.array(point_coords, ndmin=2), np.array(point_label), multimask_output=multimask_output)
        
        return self._sorted_output(masks,scores,logits)
    
    @torch.inference_mode()
    def predict_box(self, input_box:np.ndarray, image:Image.Image=None,):
        '''predict mask,score,logits given (x0y0,x1y1) box point'''
        self._update_image(image)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(point_coords=None, point_labels=None, box=np.atleast_2d(input_box), multimask_output=False)
        
        return self._sorted_output(masks,scores,logits)
    


TASK_PROMPTS = [
    "<CAPTION>", 
    "<DETAILED_CAPTION>", 
    "<MORE_DETAILED_CAPTION>", 
    "<CAPTION_TO_PHRASE_GROUNDING>", # run_example(task_prompt, text_input="A green car parked in front of a yellow building.")
    "<OD>", 
    "<DENSE_REGION_CAPTION>", 
    "<REGION_PROPOSAL>", 
    "<OCR>", 
    "<OCR_WITH_REGION>",
    '<REFERRING_EXPRESSION_SEGMENTATION>', # run_example(task_prompt, text_input="a green car")
    '<OPEN_VOCABULARY_DETECTION>', # run_example(task_prompt, text_input="a green car")
    '<REGION_TO_SEGMENTATION>', # run_example(task_prompt, text_input="<loc_702><loc_575><loc_866><loc_772>")
    '<REGION_TO_CATEGORY>', # run_example(task_prompt, text_input="<loc_52><loc_332><loc_932><loc_774>")
    '<REGION_TO_DESCRIPTION>' # run_example(task_prompt, text_input="<loc_52><loc_332><loc_932><loc_774>")
]

class Florence:
    def __init__(self, torch_dtype = torch.float16, device="cuda:0", offload:bool=True) -> None:
        # https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
        self.torch_dtype = torch_dtype
        self.device = device
        self.offload = offload
        
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).eval()#.to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

        self.generation_config = GenerationConfig(max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=3,)

    def to(self, *args, **kwargs):
        _=self.model.to(*args, **kwargs)
        return self

    def parse_texts(self, generated_texts:list[str], image:Image.Image, task:str, text_input:list[str], ):
        out_result = {}
        
        for i,generated_text in enumerate(generated_texts):
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(image.width, image.height)
            )
            
            for _task in parsed_answer:
                task_value = parsed_answer[_task]
                if isinstance(task_value,str):
                    out_result[_task] = task_value
            
                elif isinstance(task_value, dict):
                    task_result = out_result.setdefault(_task, {})
                    for k,v in task_value.items():
                        if k =='labels' and v == ['']:
                            v = [text_input[i]] # fill in label with given string
                        if 'bboxes' in k:
                            v = v[0] # [[1,2,3,4]] -> [1,2,3,4]
                        task_result.setdefault(k, []).append(v)

        return out_result


    @torch.inference_mode()
    def predict(self, image:Image.Image, task:str, text_input:str|list[str]|None=None):
        if text_input is None:
            text_input = ['']
        elif isinstance(text_input, str):
            text_input = [text_input]
        
        task_texts = [task + text for text in text_input]
        n = len(task_texts)
        
        self.to(self.device)
        
        img_tensors = torch.from_numpy(np.array(image, ndmin=4)).permute(0,3,1,2).expand(n, -1, -1, -1) # 1hwc->bchw
        inputs = self.processor(text=task_texts, images=img_tensors, return_tensors="pt", padding=True).to(self.device, self.torch_dtype)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        out_result = self.parse_texts(generated_texts, image, task, text_input)
        if self.offload:
            self.to('cpu')
            torch.cuda.empty_cache()
        return out_result
    
    @torch.inference_mode()
    def caption(self, image:Image.Image, caption_type:typing.Literal['brief','detailed','verbose']='verbose'):
        alias = {
            'brief':"<CAPTION>", 
            'detailed':"<DETAILED_CAPTION>", 
            'verbose':"<MORE_DETAILED_CAPTION>", 
        }
        task = alias[caption_type]
        return self.predict(image, task)[task]




    @staticmethod
    def draw_polygons(image:Image.Image, prediction:dict, fill_mask=False, scale = 1):  
        """  
        Draws segmentation masks with polygons on an image.  
    
        Parameters:  
        - image_path: Path to the image file.  
        - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                    'polygons' is a list of lists, each containing vertices of a polygon.  
                    'labels' is a list of labels corresponding to each polygon.  
        - fill_mask: Boolean indicating whether to fill the polygons with color.  
        """  
        colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
                    'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
        # Load the image  
        image = image.copy()
        draw = ImageDraw.Draw(image)  
        
        font = ImageFont.truetype('DejaVuSansMono.ttf', size=32)
        # Iterate over polygons and labels  
        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            if isinstance(label, list):
                label = '|'.join(label)
            color = random.choice(colormap)  
            fill_color = random.choice(colormap) if fill_mask else None  
            
            for _polygon in polygons:  
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:  
                    print('Invalid polygon:', _polygon)  
                    continue  
                
                _polygon = (_polygon * scale).reshape(-1).tolist()
                
                # Draw the polygon  
                if fill_mask:  
                    draw.polygon(_polygon, outline=color, fill=fill_color)  
                else:  
                    draw.polygon(_polygon, outline=color)
                
                # Draw the label text  
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color, font=font, stroke_width=2, stroke_fill='black')
    
        # Save or display the image  
        return image
    
    @staticmethod
    def plot_bbox(image, data):
        # Create a figure and axes  
        fig, ax = plt.subplots()  
        
        # Display the image  
        ax.imshow(image)  
        if 'bboxes_labels' in data: 
            # 'bboxes', 'bboxes_labels', 'polygons', 'polygons_labels' => 'bboxes', 'labels'
            data = {  
                'bboxes': data.get('bboxes', []) ,  
                'labels': data.get('bboxes_labels', [])    
            } 
        # Plot each bounding box  
        for bbox, label in zip(data['bboxes'], data['labels']):  
            # Unpack the bounding box coordinates  
            x1, y1, x2, y2 = bbox  
            # Create a Rectangle patch  
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
            # Add the rectangle to the Axes  
            ax.add_patch(rect)  
            # Annotate the label  
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
        
        # Remove the axis ticks and labels  
        ax.axis('off')
        return ax
    

    @staticmethod
    def poly_to_mask(img_wh:tuple[int,int], prediction:dict, flatten=False, scale = 1):  
        """  
        create segmentation masks from polygons  
    
        Parameters:  
        - img_wh: tuple of ints (width, height) 
        - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                    'polygons' is a list of lists, each containing vertices of a polygon.  
                    'labels' is a list of labels corresponding to each polygon.  
        - flatten: Boolean indicating whether to merge all masks and return a 2d array.
        """  
        
        # https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        # Set up scale factor if needed (use 1 if not scaling)  
        masks = []
        # Iterate over polygons and labels  
        for polygons in prediction['polygons']:
            mask = Image.new('L', img_wh, 0)
            for _polygon in polygons:  
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:  
                    print('Invalid polygon:', _polygon)  
                    continue  
                
                _polygon = (_polygon * scale).reshape(-1)
                
                ImageDraw.Draw(mask).polygon(_polygon.round().tolist(), outline=1, fill=1)
                # ImageDraw.Draw(mask).polygon(np.rint(_polygon).tolist(), outline=1, fill=1)
                
            masks.append(mask)
        
        masks = np.array(masks, dtype=float)
        
        if flatten:
            return masks.max(0)
        
        return masks
    



class GridHelper:
    def __init__(self, image:Image.Image, nrows: int = 8, ncols: int = 8) -> None:
        self.image = image
        self.img_width, self.img_height = image.size
        self.nrows = nrows
        self.ncols = ncols

    @property
    def cell_dims(self):
        """dimensions (w,h) of a single cell"""
        return self.img_width/self.ncols, self.img_height/self.nrows

    def _idx2row_label(self, n:int)->str:
        # only supports up to ZZ = 26*27 - 1 = 701...but that waaay to many anyway
        d,m = divmod(n,26)
        return (string.ascii_uppercase[d-1] if d>0 else '') + string.ascii_uppercase[m]
    
    def _row_label2idx(self, row_label:str) -> int:
        row = 0
        for offset,char in enumerate(row_label):
            row += string.ascii_uppercase.index(char.upper()) + offset*26
        return row
    
    def _parse_cell_label(self, label: str) -> tuple[int, int]:
        """Parse a cell label (e.g., 'A1', 'C3') into row and column indices."""
        
        match = re.match(r'([A-Z]+)(\d+)', label.upper())
        if not match:
            raise ValueError(f"Invalid cell label: {label}")
        
        row_label, col_label = match.groups()
        col = int(col_label) - 1  # Convert to 0-based index
        
        row = self._row_label2idx(row_label)
        
        return row, col
    
    def _get_cell_bounding_box(self, row: int, col: int) -> tuple[float, float, float, float]:
        """Get the bounding box for a cell."""
        cell_width,cell_height = self.cell_dims
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        return x1, y1, x2, y2

    def _get_cell_center(self, row: int, col: int, ) -> tuple[float, float]:
        """Get the center point of a cell."""
        cell_width,cell_height = self.cell_dims
        x = (col + 0.5) * cell_width
        y = (row + 0.5) * cell_height
        return x, y

    def show(self, nrows:int=None, ncols:int=None):
        if nrows is not None:
            self.nrows = nrows
        if ncols is not None:
            self.ncols = ncols
        # Create a copy of the image to draw on
        img_with_grid = self.image.copy()
        draw = ImageDraw.Draw(img_with_grid)
                
        # cell dimensions
        cell_width,cell_height = self.cell_dims
        
        # Calculate font size based on image size (adjust the divisor as needed)
        # font_size = min(width, height) // 50
        font_size = min(cell_width, cell_height) // 4
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(font_size))
        
        
        # add edge perimeter
        lw = 2
        x = self.img_width-lw # self.ncols * cell_width - lw
        y = self.img_height-lw # self.nrows * cell_height - lw
        
        draw.line([(x, 0), (x, self.img_height)], fill="white", width=lw)
        draw.line([(0, y), (self.img_width, y)], fill="white", width=lw)

        # Draw vertical lines and column labels
        for i in range(self.ncols):# + 1):
            x = i * cell_width
            draw.line([(x, 0), (x, self.img_height)], fill="white", width=lw)
            label = str(i + 1)
            bbox = font.getbbox(label)
            label_width = bbox[2] - bbox[0]
            label_x = x + (cell_width - label_width) // 2
            draw.text((label_x, 5), label, fill="white", font=font, stroke_width=1, stroke_fill='black')

        # Draw horizontal lines and row labels
        for i in range(self.nrows):
            y = i * cell_height
            draw.line([(0, y), (self.img_width, y)], fill="white", width=lw)
            label = self._idx2row_label(i)
            bbox = font.getbbox(label)
            label_height = bbox[3] - bbox[1]
            label_y = y + (cell_height - label_height) // 2
            draw.text((5, label_y), label, fill="white", font=font, stroke_width=1, stroke_fill='black')

        return img_with_grid

    def cells_to_points(self, cell_labels: list[str],) -> list[tuple[float, float]]:
        '''Convert cell labels to x,y points on the image

        Args:
            cell_labels: list of cell labels (e.g., ['A1', 'C3', ...])
        
        Returns: list cell center points
        '''
        
        results = []
        for label in cell_labels:
            row, col = self._parse_cell_label(label)
            results.append(self._get_cell_center(row, col))
        return results
    
    def cells_to_bboxes(self, cell_labels: list[str],) -> list[tuple[float, float, float, float]]:
        '''Convert cell labels to bounding boxes for the cell

        Args:
            cell_labels: list of cell labels (e.g., ['A1', 'C3', ...])
        
        Returns: list cell bounding boxes (x1, y1, x2, y2)
        '''
        results = []
        for label in cell_labels:
            row, col = self._parse_cell_label(label)
            results.append(self._get_cell_bounding_box(row, col))
        return results
    



