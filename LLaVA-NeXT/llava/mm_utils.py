from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
import numpy as np
import cv2
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
import inspect


def resize_and_center_crop(image, shortest_edge_length):
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def auto_pad_images(image, grid_params):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))

    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]

    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)
    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Step 5: Pad the resized image if necessary to match the target resolution
    pad_width = target_resolution[0] - resize_width
    pad_height = target_resolution[1] - resize_height
    padded_image = Image.new("RGB", target_resolution, color=(0, 0, 0))
    padded_image.paste(resized_image, (pad_width // 2, pad_height // 2))

    return padded_image


def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_highres_image_crop_split(image, data_args, processor=None):
    crop_resolution = data_args.image_crop_resolution
    split_resolution = data_args.image_split_resolution
    if processor is None:
        processor = data_args.image_processor
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_highres_image(image, processor, grid_pinpoints):
    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def calc_IoU(a,b):
    calc_area=lambda a :(a[2]-a[0])*(a[3]-a[1])
    a_xmin,a_ymin,a_xmax,a_ymax=a
    b_xmin,b_ymin,b_xmax,b_ymax=b
    xminmax=max(a_xmin,b_xmin)
    xmaxmin=min(a_xmax,b_xmax)
    yminmax=max(a_ymin,b_ymin)
    ymaxmin=min(a_ymax,b_ymax)
    if xminmax>=xmaxmin or yminmax>=ymaxmin:
        return 0
    s1=calc_area(a)
    s2=calc_area(b)
    sc=(xmaxmin-xminmax)*(ymaxmin-yminmax)
    #return sc/(s1+s2-sc)
    return sc/min(s1,s2) # it is not actually IoU, but this is helpful when smaller one is coverd by the bigger one


def expand_bbox(origin_bbox,target_size,image_size):
    xmin,ymin,xmax,ymax=origin_bbox
    tw,th=target_size
    w,h=image_size
    if xmax-xmin+1>=tw or ymax-ymin+1>=th:
        t=max(xmax-xmin+1,ymax-ymin+1)
        tw=th=t
    #center
    all_expand=lambda xmn,xmx,l:[(xmn-l//2,xmx+(l-l//2)),(xmn-l,xmx),(xmn,xmx+l)]
    # all_expand=lambda xmn,xmx,l:[(xmn-l//2,xmx+(l-l//2)),(xmn-l,xmx),(xmn,xmx+l)]+([(0,xmx-xmn+l)] if l-xmn>0 else [])  # 考虑了细长条的情况，有点复杂，改进也应该不大，先不加
    check_bound=lambda xmn,xmx,l:xmn>=0 and xmx<l
    
    # 如果三种情况都碰壁，则截取整张图。
    # @lyz: 但是这样会导致很严重的问题，在边缘处的异常会直接变成细长条：比如h方向不碰壁，w方向碰壁了；
    new_xmin,new_xmax=0,w-1
    new_ymin,new_ymax=0,h-1
    if xmax-xmin+1<tw:
        dl=tw-xmax+xmin-1  # delta_l: 距离目标的宽度还差多少
        for i in all_expand(xmin,xmax,dl):
            if check_bound(i[0],i[1],w):
                new_xmin,new_xmax=i
                break
    else:
        new_xmin,new_xmax=xmin,xmax
    
    if ymax-ymin+1<th:
        dl=th-ymax+ymin-1  # delta_l: 距离目标的高度还差多少
        for i in all_expand(ymin,ymax,dl):
            if check_bound(i[0],i[1],h):
                new_ymin,new_ymax=i
                break
    else:
        new_ymin,new_ymax=ymin,ymax
    return [new_xmin,new_ymin,new_xmax,new_ymax]


def gen_random_bbox(patch_size,img_size,prob_mask=None): # img size (w,h)
    if type(patch_size)!=tuple and type(patch_size)!=list:
        patch_size=(patch_size,patch_size)
    patch_size=(min(patch_size[0],img_size[0]),min(patch_size[1],img_size[1]))
    eff_size=(img_size[0]-patch_size[0]+1,img_size[1]-patch_size[1]+1)
    w,h=eff_size
    hxw=h*w
    if prob_mask is None:
        prob_mask=np.ones((h,w))/hxw
    assert prob_mask.shape[0]==h and prob_mask.shape[1]==w, "prob mask shape error"
    idx_f=np.random.choice(hxw,p=prob_mask.flatten())
    xmin=idx_f%w
    ymin=idx_f//w
    return [xmin,ymin,xmin+patch_size[0]-1,ymin+patch_size[1]-1]

def make_bbox_ready(b):
    return [b[0],b[1],b[2]+1,b[3]+1]

def get_bboxes_from_mask(mask,target_size,w,h):
    cts=cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    all_boxes=[]
    for c in cts[0]:
        xmin=c[...,0].min()
        xmax=c[...,0].max()
        ymin=c[...,1].min()
        ymax=c[...,1].max()
        op=[xmin,ymin,xmax,ymax]
        p=expand_bbox(op,(target_size,target_size),(w,h))
        flg=True
        for j in all_boxes:
            if calc_IoU(op,j)>0.8:
                flg=False
                break
            if calc_IoU(p,j)>0.1:
                flg=False
                j[0]=min(j[0],p[0])
                j[1]=min(j[1],p[1])
                j[2]=max(j[2],p[2])
                j[3]=max(j[3],p[3])
                break
        if flg:
            all_boxes.append(p)
    all_boxes=[expand_bbox(b,(1,1),(w,h)) for b in all_boxes] # make all boxes square, maybe lead to overlap boxes, find a way later
    if len(all_boxes)>4:
        all_boxes=all_boxes[:4] # should sort by score, but this one is temp for normal case oom
    return all_boxes

def process_randomroi_image(image, processor, mask=None, grid_pinpoints=None,bboxes=None, shuffle_box=False, return_bbox=False, max_box_num=4):
    """
    Process an image with mask guided patches.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        mask (PIL.Image.Image or np): A 0-1 mask shows the possible anomalies.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    w,h=image.size

    patch_size=processor.crop_size['height']
    target_size=384
    invalid=False
    if bboxes is not None:
        all_boxes=bboxes
    elif mask is None: # random crop in training
        box_num=np.random.randint(1,max_box_num+1)
        all_boxes=[gen_random_bbox((target_size,target_size),(w,h)) for i in range(box_num)]

    else:
        if not isinstance(mask,np.ndarray):
            mask=np.array(mask)
            if mask.shape[0]!=h or mask.shape[1]!=w:
                mask=cv2.resize(mask,(w,h))
        else: #persume np array mask is normalized 0-1
            if mask.shape[0]!=h or mask.shape[1]!=w:
                mask=cv2.resize(mask,(w,h)) # should not use cv2 resize, but for now it is convenient
            mn_score,mx_score=mask.min(),mask.max()
            dscore=(mx_score-mn_score)*0.9+mn_score
            _,mask=cv2.threshold(mask,dscore,255,cv2.THRESH_BINARY)
            mask=mask.astype(np.uint8)
        all_boxes=get_bboxes_from_mask(mask,target_size,w,h)

    # 对box进行shuffle
    if shuffle_box:
        # print("Shuffle box!")
        np.random.shuffle(all_boxes)

    if len(all_boxes)>max_box_num:
        all_boxes=all_boxes[:max_box_num]
        print("remove extra boxes >",max_box_num)
    assert len(all_boxes)<=max_box_num, f"too many patches for now {len(all_boxes)}"

    # anyres part
    assert grid_pinpoints is not None
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))

    extra_patches = list([image.crop(make_bbox_ready(box)).resize((target_size,target_size)) for box in all_boxes])
    image_patches = [image_original_resize] + extra_patches

    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    if return_bbox:
        return torch.stack(image_patches, dim=0),all_boxes
    return torch.stack(image_patches, dim=0)#.rename("mask_patches","C","H","W")  N, 3, 336, 336


def process_anyres_max_9_randomroi_image(image, processor, mask=None, grid_pinpoints=None,bboxes=None, shuffle_box=False, return_bbox=False):
  
    anyres_patches=process_anyres_image(image, processor, grid_pinpoints)
    randomroi_patches=process_randomroi_image(image, processor, mask, grid_pinpoints, bboxes, shuffle_box, return_bbox)
    if return_bbox:
        randomroi_patches,bbox=randomroi_patches
    else:
        bbox=None
    total_patches=torch.concat([anyres_patches,randomroi_patches[1:]],dim=0)
    #print("mm_utils",anyres_patches.shape,randomroi_patches.shape,total_patches.shape)
    if bbox is not None:
        return total_patches,bbox
    return total_patches

def process_randomroi_wrapper(image_aspect_ratio,*args,**kwargs):
    #print("wrapper report",image_aspect_ratio)
    process_func=process_randomroi_image if image_aspect_ratio=="randomroi" else process_anyres_max_9_randomroi_image
    param_name=inspect.signature(process_func).parameters
    kwargs={k:kwargs[k] for k in kwargs if k in param_name}
    return process_func(*args,**kwargs)

def process_images(images, image_processor, model_cfg, masks=None, boxes_list=None, overwrite_img_asp=None):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    if overwrite_img_asp is not None:
        print(f"overwrite process image with {overwrite_img_asp}")
        image_aspect_ratio = overwrite_img_asp
    else:
        print(f"process image with {image_aspect_ratio} from model config")
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    ######################randomroi############################
    elif "randomroi" in image_aspect_ratio:
        process_func=process_randomroi_image if image_aspect_ratio=="randomroi" else process_anyres_max_9_randomroi_image
        # print(image_aspect_ratio)
        if boxes_list is not None:
            for image,bboxes in zip(images,boxes_list):
                image = process_func(image, image_processor, None, model_cfg.image_grid_pinpoints,bboxes)
                new_images.append(image)
        else:
            if masks is None:
                masks=[None]*len(images)
            if len(masks)!=len(images):
                dl=len(images)-len(masks)
                masks=masks+[None]*dl
            box_tmp=[]
            for idx,(image,mask) in enumerate(zip(images,masks)):
                if type(mask) is str: #use ref_bbox to let the ref img use same random bbox as Q
                    assert mask=="ref_bbox", "only ref_bbox are supported as mask flag, but got mask: "+mask
                    assert idx!=0, "ref idx can not be 0"
                    ref_bbox=box_tmp[idx-1]
                    image,bbox = process_func(image, image_processor, None, model_cfg.image_grid_pinpoints,bboxes=ref_bbox,return_bbox=True)
                else:
                    image,bbox = process_func(image, image_processor, mask, model_cfg.image_grid_pinpoints,return_bbox=True)
                box_tmp.append(bbox)
                new_images.append(image)
    #########################################################
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
