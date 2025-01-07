import time 
import contextlib
import collections
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T 

import tensorrt as trt
import os
from torch.utils.data import Dataset, DataLoader
import json
from timeit import default_timer as timer

class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        '''build binddings
        '''
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        # max_batch_size = 1

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
            
            # TODO (lyuwenyu): check dtype, 
            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)
            # if self.bindings[n].data.dtype != blob[n].shape:
            #     blob[n] = blob[n].to(self.bindings[n].data.dtype)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs


    def async_run_cuda(self, blob):
        '''numpy input
        '''
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    
    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)

        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)

    def synchronize(self, ):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

        elif self.backend == 'cuda':
            self.stream.synchronize()
    
    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 


    @staticmethod
    def onnx2tensorrt():
        pass

def get_data(json_data):
    images, annotations = json_data["images"], json_data["annotations"]
    images_dict = {}
    for i in images:
        images_dict[i["id"]] = {"file_name": i["file_name"], "bbox": []}

    for ann in annotations:
        images_dict[ann["image_id"]]["bbox"].append(ann["bbox"])

    return images_dict

class ImageDataset(Dataset):
    def __init__(self, json_file, image_root_dir, transform=None):
        self.transform = transform
        with open(json_file) as f:
            json_data = json.load(f)

        images_dict = get_data(json_data)
        self.images = list(images_dict.values())
        self.image_root_dir = image_root_dir

    def __len__(self):
        print(len(self.images))
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        filename = info["file_name"]
        file_path = os.path.join(self.image_root_dir, filename)
        image = Image.open(file_path).convert("RGB")
        orig_size = torch.tensor(image.size[::-1])
        if self.transform:
            image = self.transform(image)
        return image, file_path, orig_size

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', type=str, default="/colon_workspace/RT-DETR/rtdetrv2_pytorch/rt_detr_dynamic.trt")
    parser.add_argument('-f', '--im-folder', type=str, default="/colon_workspace/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_images")
    parser.add_argument('-a', '--ann-file', type=str, default="/colon_workspace/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_ann.json")
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-b', '--batch-size', type=int, default=128)

    args = parser.parse_args()

    model = TRTInference(args.trt_file, device=args.device)
    test_ann = args.ann_file
    test_path = args.im_folder
    batch_size = args.batch_size

    # Define the image transformations
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    dataset = ImageDataset(json_file=test_ann, image_root_dir=test_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    start = timer()

    for batch_images, image_paths, orig_sizes in dataloader:
        batch_images = batch_images.to(args.device)
        orig_sizes = orig_sizes.to(args.device)
        blob = {
            'images': batch_images,
            'orig_target_sizes': orig_sizes,
        }
        output = model(blob)

    end = timer()
    print(f"Time taken: {end - start} seconds")
