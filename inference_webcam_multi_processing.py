
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## todo

import cv2
import time
import torch
import colorsys
import numpy as np
from multi_person_tracker import MPT
from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.demo_utils import (
    prepare_rendering_results,
    convert_crop_cam_to_orig_img,
    download_ckpt)
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from yolov3.yolo import YOLOv3
from multi_person_tracker import Sort
from lib.data_utils.img_utils import get_single_image_crop_demo
from collections import OrderedDict
#from multiprocessing import Process, Queue, freeze_support, set_start_method
from torch.multiprocessing import Queue, Process, set_start_method
from queue import Empty as QueueEmpty
import multiprocessing

class MPT():
    def __init__(
            self,
            device=None,
            batch_size=12,
            display=False,
            detection_threshold=0.7,
            detector_type='yolo',
            yolo_img_size=608,
            output_format='list',
    ):
        '''
        Multi Person Tracker

        :param device (str, 'cuda' or 'cpu'): torch device for model and inputs
        :param batch_size (int): batch size for detection model
        :param display (bool): display the results of multi person tracking
        :param detection_threshold (float): threshold to filter detector predictions
        :param detector_type (str, 'maskrcnn' or 'yolo'): detector architecture
        :param yolo_img_size (int): yolo detector input image size
        :param output_format (str, 'dict' or 'list'): result output format
        '''

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.display = display
        self.detection_threshold = detection_threshold
        self.output_format = output_format

        if detector_type == 'maskrcnn':
            self.detector = keypointrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        elif detector_type == 'yolo':
            self.detector = YOLOv3(
                device=self.device, img_size=yolo_img_size, person_detector=True, video=True, return_dict=True
            )
        else:
            raise ModuleNotFoundError

        self.tracker = Sort()

    @torch.no_grad()
    def run_tracker(self, frame):
        '''
        Run tracker on an input video

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        # initialize tracker
        #self.tracker = Sort()

        start = time.time()
        print('Running Multi-Person-Tracker')
        trackers = []
        frame = frame.to(self.device)
        #print('detector frame',frame.shape)
        predictions = self.detector(frame)
        for pred in predictions:
            bb = pred['boxes'].cpu().numpy()
            sc = pred['scores'].cpu().numpy()[..., None]
            dets = np.hstack([bb,sc])
            dets = dets[sc[:,0] > self.detection_threshold]
            #print('dets222222   ',dets)
            # if nothing detected do not update the tracker
            if dets.shape[0] > 0:
                track_bbs_ids = self.tracker.update(dets)
            else:
                track_bbs_ids = np.empty((0, 5))
            trackers.append(track_bbs_ids)

        runtime = time.time() - start
        fps = 1 / runtime
        print(f'Finished. Detection + Tracking FPS {fps:.2f}')
        return trackers

    def prepare_output_tracks(self, trackers):
        '''
        Put results into a dictionary consists of detected people
        :param trackers (ndarray): input tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: dict: of people. each key represent single person with detected bboxes and frame_ids
        '''
        people = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])
                # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
            people[k]['frames'] = np.array(people[k]['frames'])

        return people

    def __call__(self, frame, output_file=None):
        '''
        Execute MPT and return results as a dictionary of person instances

        :param video (ndarray): input video tensor of shape NxHxWxC
        :return: a dictionary of person instances
        '''


        trackers = self.run_tracker(frame)
        #print('tracker: ',trackers)
        if self.display:
            self.display_results(frame, trackers, output_file)

        if self.output_format == 'dict':
            result = self.prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers

        return result

MIN_NUM_FRAMES = 25


def Inference(frame, bboxes=None, scale=1.0, crop_size=224):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    norm_img, raw_img, kp_2d = get_single_image_crop_demo(
        img,
        bboxes[0],
        kp_2d=None,
        scale=scale,
        crop_size=crop_size)
    #norm_img = norm_img/255. #TODO 可能会有多张图片
    return norm_img

def prepare_rendering_results(vibe_results, nframes): # TODO cam的参数只有一个
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
            }
    #print('frame_results00000000000',frame_results)
    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results): # TODO 不懂是什么意思
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        #print('sort_idx',sort_idx)
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

def render(frame_orig, vibe_results,renderer):
    render_time = time.time()
    # load renderer
    #renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)
    # prepare results for rendering
    num_frames = 1
    #print('vibe_results1111: ',vibe_results)
    #vibe_results[1]['orig_cam'] = vibe_results[1]['orig_cam'][np.newaxis,:]
    #print('orig_cam:   ',vibe_results[1]['orig_cam'].shape)
    frame_results = prepare_rendering_results(vibe_results, num_frames)
    #print('frame_results',frame_results)
    img = frame_orig
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    for person_id, person_data in frame_results[0].items():
        frame_verts = person_data['verts']
        frame_cam = person_data['cam']
        #print('4444444444frame_cam',frame_cam)
        mc = mesh_color[person_id]

        mesh_filename = None

        img = renderer.render(
            img,
            frame_verts,
            cam=frame_cam,
            color=mc,
            mesh_filename=mesh_filename,
        )
    fps = 1 / (time.time() - render_time)
    print(f'RENDER FPS: {fps:.2f}')
    return img

def worker(input, output):
    for frame_orig, vibe_results in iter(input.get, 'STOP'):
        result = render(frame_orig, vibe_results)
        print(result.shape)
        output.put(result)

def detect_track_vibe(frame_orig, model_dettr, model_vibe, device):
    orig_height, orig_width = frame_orig.shape[:2]
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    frame = frame / 255.
    frame = frame.transpose((2, 0, 1))
    frame = torch.from_numpy(frame)
    frame = frame.unsqueeze(0)
    tracking_results = model_dettr(frame)
    # print('1111111111111tracking result',tracking_results)

    # print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in list(tracking_results.keys()):
        bboxes = joints2d = None

        bboxes = tracking_results[person_id]['bbox']  # shape(1,4)
        # print('bboxes:  ',bboxes) #相同

        frames = tracking_results[person_id]['frames']
        # print('22222222',bboxes)
        bbox_scale = 1.1
        dataset = Inference(frame=frame_orig, bboxes=bboxes, scale=bbox_scale)

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            batch = dataset
            batch = batch.unsqueeze(0).unsqueeze(0)
            batch = batch.to(device)
            # print(batch.shape)
            batch_size, seqlen = batch.shape[:2]
            output = model_vibe(batch)[-1]

            pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
            pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
            pred_pose.append(output['theta'][:, :, 3:75].reshape(batch_size * seqlen, -1))
            pred_betas.append(output['theta'][:, :, 75:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        pred_cam = pred_cam.cpu().numpy()
        # print('pred_cam:  ',pred_cam)  #不同
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        # print('3333333333',pred_cam)
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )
        # print('orig_cam',orig_cam.shape)
        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }
        vibe_results[person_id] = output_dict
    # print('vibe_results orig_cam:  ',vibe_results[1]['orig_cam'])
    # print('vibe_results pose:  ', vibe_results[1]['pose'])
    end = time.time()
    fps = 1 / (end - vibe_time)
    print(f'VIBE FPS: {fps:.2f}')
    return vibe_results

def putter(model_1_ready,model_2_ready, queue_img):
    #time.sleep(50)
    cap = cv2.VideoCapture('sample_video.mp4')
    i = 0
    while True:
        if (model_1_ready.value==3 and model_2_ready.value==5): #TODO
            i += 1
            print('frameaaaaaaaaaaaaa:        ',i)
            ret, frame_orig = cap.read()
            if frame_orig is None:
                break
            frame_orig = np.array(frame_orig)
            queue_img.put(frame_orig)
    print('finish')

def get_put(model_1_ready,queue_img, queue_dettra,total_time):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot = MPT(
        device=device,
        batch_size=1,
        output_format='dict'
    )
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    model_1_ready.value += 1
    total_time.value = time.time()
    i = 0
    while True:
        try:
            total_time = time.time()
            frame_orig = queue_img.get(True, 50)
            # block为True,就是如果队列中无数据了。
            #   |—————— 若timeout默认是None，那么会一直等待下去。
            #   |—————— 若timeout设置了时间，那么会等待timeout秒后才会抛出Queue.Empty异常
            # block 为False，如果队列中无数据，就抛出Queue.Empty异常
            i += 1
            print('dettra22222',i)
            vibe_results = detect_track_vibe(frame_orig, mot, model, device)
            queue_dettra.put((frame_orig, vibe_results))
            total_time = time.time() - total_time
            print(f'Detection+Tracking FPS : {1 / total_time:.2f}.')
        except QueueEmpty:
            break

def getter(model_2_ready, queue_dettra):
    renderer = Renderer(resolution=(1920, 1080), orig_img=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/1.avi', fourcc, 30.0, (1920, 1080), True)
    model_2_ready.value += 1
    i = 1
    while True:
        try:
            frame_orig, vibe_results = queue_dettra.get(True, 50)
            # block为True,就是如果队列中无数据了。
            #   |—————— 若timeout默认是None，那么会一直等待下去。
            #   |—————— 若timeout设置了时间，那么会等待timeout秒后才会抛出Queue.Empty异常
            # block 为False，如果队列中无数据，就抛出Queue.Empty异常
            i += 1
            print('render333333', i)
            img = render(frame_orig, vibe_results,renderer)
            out.write(img)
        except QueueEmpty:
            print('================= END =================')
            break
def tmpfunc(name,name1):
    print(name)

if __name__ == '__main__':

    set_start_method('spawn',force=True) #success
    model_1_ready = multiprocessing.Value('d', 0)
    model_2_ready = multiprocessing.Value('d', 0)
    total_time = multiprocessing.Value('d', 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ========= Run tracking ========= #
    '''
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=1,
        output_format='dict'
    )
    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    
    renderer = Renderer(resolution=(1920, 1080), orig_img=True)
    '''
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #cap = cv2.VideoCapture('test.avi')
    #out = cv2.VideoWriter('output/test_5people.avi', fourcc, 30.0, (640, 360), True)
    #cap = cv2.VideoCapture('sample_video.mp4')
    #out = cv2.VideoWriter('output/1.avi', fourcc, 30.0, (1920, 1080), True)
    #freeze_support()

    NUMBER_OF_PROCESSES_dettra = 3
    NUMBER_OF_PROCESSES_render = 5
    # load renderer


    queue_1 = Queue()
    queue_2 = Queue()
    process_dettr = []
    process_render = []
    putter_process = Process(target=putter, args=(model_1_ready,model_2_ready, queue_1))
    for i in range(NUMBER_OF_PROCESSES_dettra):
        process_dettr.append(Process(target=get_put, args=(model_1_ready,queue_1,queue_2,total_time)))
    for i in range(NUMBER_OF_PROCESSES_render):
        process_render.append(Process(target=getter, args=(model_2_ready, queue_2)))
    for i in range(NUMBER_OF_PROCESSES_dettra):
        a = process_dettr[i]
        a.start()
    for i in range(NUMBER_OF_PROCESSES_render):
        a = process_render[i]
        a.start()
    #success
    print(111111111111111111111)
    putter_process.start()
    #putter_process.join()

    for i in range(NUMBER_OF_PROCESSES_dettra):
        a = process_dettr[i]
        a.join()
    print(33333333333333333333)
    print('FPS final:  ',301/(time.time()- total_time.value-50))