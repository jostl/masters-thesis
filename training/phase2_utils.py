import torch
import numpy as np
import random
import augmenter
from torchvision import transforms
import torchvision.transforms.functional as TF

import sys
import glob
try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError as e:
    pass
import cv2
import utils.carla_utils as cu
from models.image import ImagePolicyModelSS, FullModel
from models.birdview import BirdViewPolicyModelSS
from perception.utils.helpers import get_segmentation_tensor
from perception.utils.visualization import display_images_horizontally, get_rgb_segmentation, get_segmentation_colors
from perception.utils.segmentation_labels import DEFAULT_CLASSES
from bird_view.models import common
CROP_SIZE = 192
PIXELS_PER_METER = 5

from pathlib import Path
def repeat(a, repeats, dim=0):
    """
    Substitute for numpy's repeat function. Taken from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(
            torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)


def get_weight(learner_points, teacher_points):
    decay = torch.FloatTensor([0.7**i for i in range(5)]).to(learner_points.device)
    xy_bias = torch.FloatTensor([0.7,0.3]).to(learner_points.device)
    loss_weight = torch.mean((torch.abs(learner_points - teacher_points)*xy_bias).sum(dim=-1)*decay, dim=-1)
    x_weight = torch.max(
        torch.mean(teacher_points[...,0],dim=-1),
        torch.mean(teacher_points[...,0]*-1.4,dim=-1),
    )
    
    return loss_weight

def weighted_random_choice(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t, np.random.uniform(0,s))

def get_optimizer(parameters, lr=1e-4):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer

def load_image_model(backbone, ckpt, device='cuda'):
    net = ImagePolicyModelSS(
        backbone,
        all_branch=True
    ).to(device)
    
    net.load_state_dict(torch.load(ckpt))
    return net
    
def _log_visuals(rgb_image, birdview, speed, command, loss, pred_locations, _pred_locations, _teac_locations, size=16):
    import cv2
    import numpy as np
    import utils.carla_utils as cu

    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]
    _numpy = lambda x: x.detach().cpu().numpy().copy()

    images = list()

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        canvas = np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy()
        canvas = cu.visualize_birdview(canvas)
        rgb = np.uint8(_numpy(rgb_image[i]).transpose(1, 2, 0) * 255).copy()
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

        def _write(text, i, j):
            cv2.putText(
                    canvas, text, (cols[j], rows[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        def _dot(_canvas, i, j, color, radius=2):
            x, y = int(j), int(i)
            _canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color
        
        def _stick_together(a, b):
            h = min(a.shape[0], b.shape[0])
    
            r1 = h / a.shape[0]
            r2 = h / b.shape[0]
    
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
    
            return np.concatenate([a, b], 1)
        
        _command = {
                1: 'LEFT', 2: 'RIGHT',
                3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item()+1, '???')

        _dot(canvas, 0, 0, WHITE)

        for x, y in (_teac_locations[i] + 1) * (0.5 * CROP_SIZE): _dot(canvas, x, y, BLUE)
        for x, y in _pred_locations[i]: _dot(rgb, x, y, RED)
        for x, y in pred_locations[i]: _dot(canvas, x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)
        
        
        images.append((loss[i].item(), _stick_together(rgb, canvas)))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]
    
def load_birdview_model(backbone, ckpt, device='cuda'):
    teacher_net = BirdViewPolicyModelSS(backbone, all_branch=True).to(device)
    teacher_net.load_state_dict(torch.load(ckpt))
    
    return teacher_net
    
class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=1.4, fixed_offset=4.0, device='cuda'):
        self._img_size = torch.FloatTensor([w,h]).to(device)
        
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        print ("Fixed offset", fixed_offset)
    
    def __call__(self, camera_locations):
        if isinstance(camera_locations, torch.Tensor):
            camera_locations = (camera_locations + 1) * self._img_size/2
        else:
            camera_locations = (camera_locations + 1) * self._img_size.cpu().numpy()/2
        
        w, h = self._img_size
        w = int(w)
        h = int(h)
        
        cx, cy = w/2, h/2

        f = w /(2 * np.tan(self._fov * np.pi / 360))
    
        xt = (camera_locations[...,0] - cx) / f
        yt = (camera_locations[...,1] - cy) / f

        world_z = self._world_y / yt
        world_x = world_z * xt
        
        if isinstance(camera_locations, torch.Tensor):
            map_output = torch.stack([world_x, world_z],dim=-1)
        else:
            map_output = np.stack([world_x,world_z],axis=-1)
    
        map_output *= PIXELS_PER_METER
        map_output[...,1] = CROP_SIZE - map_output[...,1]
        map_output[...,0] += CROP_SIZE/2
        map_output[...,1] += self._fixed_offset*PIXELS_PER_METER
        
        return map_output

class LocationLoss(torch.nn.Module):
    def forward(self, pred_locations, teac_locations):
        pred_locations = pred_locations/(0.5*CROP_SIZE) - 1
        
        return torch.mean(torch.abs(pred_locations - teac_locations), dim=(1,2,3))

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, buffer_limit=100000, augment=None, sampling=True, aug_fix_iter=1000000, batch_aug=4,
                 use_cv=False, semantic_classes=DEFAULT_CLASSES):
        self.buffer_limit = buffer_limit
        self._data = []
        self._weights = []
        self.rgb_transform = transforms.ToTensor()
        
        self.birdview_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.normalize_rgb = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        if augment and augment != 'None':
            self.augmenter = getattr(augmenter, augment)
        else:
            self.augmenter = None
            
        self.normalized = False
        self._sampling = sampling
        self.aug_fix_iter = aug_fix_iter
        self.batch_aug = batch_aug

        self.use_cv = use_cv
        self.semantic_classes = semantic_classes
            
    def __len__(self):
        return len(self._data)

    def __getitem__(self, _idx):
        if self._sampling and self.normalized:
            while True:
                idx = weighted_random_choice(self._weights)
                if idx < len(self._data):
                    break
                print ("waaat")
        else:
            idx = _idx
            
        image_data, cmd, speed, target, birdview_img = self._data[idx]
        if self.use_cv:
            rgb_img = image_data[0]
        else:
            rgb_img = image_data

        if self.augmenter:
            rgb_imgs = [self.augmenter(self.aug_fix_iter).augment_image(rgb_img) for i in range(self.batch_aug)]
        else:
            rgb_imgs = [rgb_img for i in range(self.batch_aug)]

        rgb_imgs = [self.rgb_transform(img) for img in rgb_imgs]
        if self.batch_aug == 1:
            rgb_imgs = rgb_imgs[0]
        else:
            rgb_imgs = torch.stack(rgb_imgs)

        birdview_img = self.birdview_transform(birdview_img)

        if self.use_cv:
            semantic_image = self.to_tensor(get_segmentation_tensor(image_data[1], classes=DEFAULT_CLASSES)).float()
            depth_image = self.to_tensor(image_data[2] / 255 ).float()
            rgb_imgs = self.normalize_rgb(rgb_imgs)
            image = torch.cat([rgb_imgs, semantic_image, depth_image])
            return idx, image, cmd, speed, target, birdview_img
        return idx, rgb_imgs, cmd, speed, target, birdview_img
        
    def update_weights(self, idxes, losses):
        idxes = idxes.numpy()
        losses = losses.detach().cpu().numpy()
        for idx, loss in zip(idxes, losses):
            if idx > len(self._data):
                continue

            self._new_weights[idx] = loss
            
    def init_new_weights(self):
        self._new_weights = self._weights.copy()
            
    def normalize_weights(self):
        self._weights = self._new_weights
        self.normalized = True

    def add_data(self, rgb_img, cmd, speed, target, birdview_img, weight, semseg=None, depth=None):
        self.normalized = False
        if semseg is not None and depth is not None:
            image_data = (rgb_img, semseg, depth)
        else:
            image_data = rgb_img
        self._data.append((image_data, cmd, speed, target, birdview_img))
        self._weights.append(weight)
            
        if len(self._data) > self.buffer_limit:
            # Pop the one with lowest loss
            idx = np.argsort(self._weights)[0]
            self._data.pop(idx)
            self._weights.pop(idx)
            
            
    def remove_data(self, idx):
        self._weights.pop(idx)
        self._data.pop(idx)
            
    def get_highest_k(self, k):
        top_idxes = np.argsort(self._weights)[-k:]
        images = []
        bird_views = []
        targets = []
        cmds = []
        speeds = []
        
        for idx in top_idxes:
            if idx < len(self._data):
                image, cmd, speed, target, birdview_img = self._data[idx]
                if self.use_cv:
                    def transpose(img):
                        return img.transpose(2, 0, 1)
                    rgb_img, semseg, depth = image
                    rgb_img = self.normalize_rgb(self.rgb_transform(rgb_img)).numpy()
                    semseg = transpose(get_segmentation_tensor(semseg, classes=DEFAULT_CLASSES))
                    depth = np.array([depth]) / 255
                    image = np.concatenate((rgb_img, semseg, depth), axis=0).transpose(1, 2, 0)

                images.append(TF.to_tensor(np.ascontiguousarray(image)))
                bird_views.append(TF.to_tensor(birdview_img))
                cmds.append(cmd)
                speeds.append(speed)
                targets.append(target)

        return torch.stack(images), torch.stack(bird_views), torch.FloatTensor(cmds), torch.FloatTensor(
            speeds), torch.FloatTensor(targets)

    def get_weights(self):
        return self._weights

    def get_image_data(self):
        image_data = [self._data[i][0:-1] for i in range(len(self._data))]
        return image_data

    def get_birdview_data(self):
        birdview_data = [self._data[i][-1] for i in range(len(self._data))]
        return birdview_data


class ReplayBufferDisk(torch.utils.data.Dataset):
    def __init__(self, path, buffer_limit=100000, augment=None, sampling=True, aug_fix_iter=1000000, batch_aug=4,
                 use_cv=False, semantic_classes=DEFAULT_CLASSES, **kwargs):
        self.buffer_limit = buffer_limit
        self._data = []
        self._weights = []
        self.rgb_transform = transforms.ToTensor()

        self.birdview_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.normalize_rgb = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        if augment and augment != 'None':
            self.augmenter = getattr(augmenter, augment)
        else:
            self.augmenter = None

        self.normalized = False
        self._sampling = sampling
        self.aug_fix_iter = aug_fix_iter
        self.batch_aug = batch_aug

        self.use_cv = use_cv
        self.semantic_classes = semantic_classes
        # Folder path
        self.path = Path(path)
        self.image_nr = 0

        if not self.path.exists():
            self.path.mkdir()
        self.rgb_folder = self.path / "rgb"
        self.birdview_folder = self.path / "birdview"
        paths = [self.rgb_folder, self.birdview_folder]
        if self.use_cv:
            self.segmentation_folder = self.path / "segmentation"
            self.depth_folder = self.path / "depth"
            paths.append(self.segmentation_folder)
            paths.append(self.depth_folder)
        for path in paths:
            if not path.exists():
                path.mkdir()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, _idx):
        if self._sampling and self.normalized:
            while True:
                idx = weighted_random_choice(self._weights)
                if idx < len(self._data):
                    break
                print("waaat")
        else:
            idx = _idx

        image_paths, cmd, speed, target, birdview_path = self._data[idx]

        if self.use_cv:
            rgb_path, semseg_path, depth_path = image_paths
            semseg_numpy = self.read_rgb(semseg_path)
            depth_numpy = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 255
        else:
            rgb_path = image_paths
        rgb_img = self.read_rgb(rgb_path)

        if self.augmenter:
            rgb_imgs = [self.augmenter(self.aug_fix_iter).augment_image(rgb_img) for i in range(self.batch_aug)]
        else:
            rgb_imgs = [rgb_img for i in range(self.batch_aug)]

        rgb_imgs = [self.rgb_transform(img) for img in rgb_imgs]
        if self.batch_aug == 1:
            rgb_imgs = rgb_imgs[0]
        else:
            rgb_imgs = torch.stack(rgb_imgs)

        birdview_img = self.birdview_transform(torch.load(birdview_path))

        if self.use_cv:
            semantic_image = self.to_tensor(get_segmentation_tensor(semseg_numpy, classes=DEFAULT_CLASSES)).float()
            depth_image = self.to_tensor(depth_numpy).float()
            rgb_imgs = self.normalize_rgb(rgb_imgs)
            image = torch.cat([rgb_imgs, semantic_image, depth_image])
            return idx, image, cmd, speed, target, birdview_img
        return idx, rgb_imgs, cmd, speed, target, birdview_img

    def read_rgb(self, img_path):
        return cv2.imread(img_path)

    def update_weights(self, idxes, losses):
        idxes = idxes.numpy()
        losses = losses.detach().cpu().numpy()
        for idx, loss in zip(idxes, losses):
            if idx > len(self._data):
                continue

            self._new_weights[idx] = loss

    def init_new_weights(self):
        self._new_weights = self._weights.copy()

    def normalize_weights(self):
        self._weights = self._new_weights
        self.normalized = True

    def add_data(self, rgb_img, cmd, speed, target, birdview_img, weight, semseg=None, depth=None):
        self.normalized = False
        file = str(self.image_nr) + ".png"
        rgb_path = str(self.rgb_folder / file)
        if semseg is not None and depth is not None:
            semseg_path = str(self.segmentation_folder / file)
            depth_path = str(self.depth_folder / file)
            image_path = (rgb_path, semseg_path, depth_path)

            cv2.imwrite(rgb_path, rgb_img)
            cv2.imwrite(semseg_path, semseg)
            cv2.imwrite(depth_path, depth)
        else:
            image_path = rgb_path
            cv2.imwrite(rgb_path, rgb_img)
        birdview_path = str(self.birdview_folder / str(str(self.image_nr) + ".birdview"))
        torch.save(birdview_img, birdview_path)

        self.image_nr += 1
        self._data.append((image_path, cmd, speed, target, birdview_path))
        self._weights.append(weight)

        if len(self._data) > self.buffer_limit:
            # Pop the one with lowest loss
            idx = np.argsort(self._weights)[0]
            self._data.pop(idx)
            self._weights.pop(idx)

    def remove_data(self, idx):
        self._weights.pop(idx)
        self._data.pop(idx)

    def get_highest_k(self, k):
        top_idxes = np.argsort(self._weights)[-k:]
        images = []
        bird_views = []
        targets = []
        cmds = []
        speeds = []

        for idx in top_idxes:
            if idx < len(self._data):
                image_path, cmd, speed, target, birdview_path = self._data[idx]
                if self.use_cv:
                    def transpose(img):
                        return img.transpose(2, 0, 1)

                    rgb_path, semseg_path, depth_path = image_path
                    rgb_img = self.normalize_rgb(self.rgb_transform(self.read_rgb(rgb_path))).numpy()
                    semseg = transpose(get_segmentation_tensor(self.read_rgb(semseg_path),
                                                               classes=DEFAULT_CLASSES))
                    depth = np.array([cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)]) / 255
                    image = np.concatenate((rgb_img, semseg, depth), axis=0).transpose(1, 2, 0)
                else:
                    image = self.read_rgb(image_path)
                birdview_img = torch.load(birdview_path)
                images.append(TF.to_tensor(np.ascontiguousarray(image)))
                bird_views.append(TF.to_tensor(birdview_img))
                cmds.append(cmd)
                speeds.append(speed)
                targets.append(target)

        return torch.stack(images), torch.stack(bird_views), torch.FloatTensor(cmds), torch.FloatTensor(
            speeds), torch.FloatTensor(targets)

def setup_image_model(backbone, imagenet_pretrained, device,  semantic_classes=DEFAULT_CLASSES,
                      image_ckpt="", use_cv=False, trained_cv=False, all_branch=False, return_cv_preds=True, **kwargs):
    if trained_cv:
        net = FullModel(backbone=backbone, image_pretrained=imagenet_pretrained, all_branch=all_branch,
                        image_ckpt=image_ckpt)
        net.return_cv_preds = return_cv_preds
    elif use_cv:
        net = ImagePolicyModelSS(backbone, pretrained=imagenet_pretrained, all_branch=all_branch,
                                 input_channel=len(semantic_classes) + 5)
        if image_ckpt:
            net.load_state_dict(torch.load(image_ckpt))
    else:
        net = ImagePolicyModelSS(
            backbone,
            pretrained=imagenet_pretrained,
            all_branch=all_branch
        )
        if image_ckpt:
            net.load_state_dict(torch.load(image_ckpt))
    return net.to(device)

def show_rgb_semseg_depth(image):
    rgb = image[0, 0:3]
    semseg = image[0, 3:12]
    depth = image[0, -1]

    display_images = [rgb.cpu().numpy().transpose(1, 2, 0),
                      semseg.cpu().numpy().transpose(1, 2, 0),
                      depth.cpu().numpy()]
    class_colors = get_segmentation_colors(9, class_indxs=DEFAULT_CLASSES)
    semseg_rgb = get_rgb_segmentation(semantic_image=display_images[1],
                                      class_colors=class_colors)

    semseg_rgb = semseg_rgb / 255
    display_images[1] = semseg_rgb

    display_images_horizontally(display_images, fig_width=10, fig_height=2)
