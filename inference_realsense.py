
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['EGL_DEVICE_ID'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## todo
import pyrealsense2 as rs
import cv2
import time
import torch
import colorsys
import argparse
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
'''
def prepare_rendering_results(vibe_results, nframes):
    frame_results = {}
    for person_id, person_data in vibe_results.items():
        frame_results[person_id] = {
            'verts': person_data['verts'],
            'cam': person_data['orig_cam'],
        }
    # naive depth ordering based on the scale of the weak perspective camera

    sort_idx = np.argsort([v['cam'] for k,v in frame_results.items()])
    frame_results = OrderedDict(
        {list(frame_results.keys())[i]:frame_results[list(frame_results.keys())[i]] for i in sort_idx}
    )
    return frame_results
'''
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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    # ========= Run tracking ========= #
    bbox_scale = 1.1
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
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


    # ========= main loop ======================= #
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #cap = cv2.VideoCapture('test.avi')
    #out = cv2.VideoWriter('output/test_5people.avi', fourcc, 30.0, (640, 360), True)
    #cap = cv2.VideoCapture('sample_video.mp4')
    out = cv2.VideoWriter('output/test_realsense_2.avi', fourcc, 10.0, (640, 480), True)

    # load renderer
    renderer = Renderer(resolution=(640, 480), orig_img=True, wireframe=args.wireframe)

    i = 0
    time_acc = 0.0
    while (True):
        # Capture frame-by-frame
        total_time = time.time()
        frames = pipeline.wait_for_frames()
        frame_orig = frames.get_color_frame()

        # Convert images to numpy arrays
        frame_orig = np.asanyarray(frame_orig.get_data())

        #ret, frame_orig = cap.read()
        if frame_orig is None:
            break
    #for i in range(1,300):
    #    total_time = time.time()
    #    path = os.path.join('tmp/sample_video/',f'{i:06d}.png')
    #    frame_orig = cv2.imread(path)
        orig_height, orig_width = frame_orig.shape[:2]
        frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        frame = frame/255.
        frame = frame.transpose((2,0,1))
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0)
        tracking_results = mot(frame)
        #print('1111111111111tracking result',tracking_results)

        #print(f'Running VIBE on each tracklet...')
        vibe_time = time.time()
        vibe_results = {}
        for person_id in list(tracking_results.keys()):
            bboxes = joints2d = None

            bboxes = tracking_results[person_id]['bbox'] # shape(1,4)
            #print('bboxes:  ',bboxes) #相同

            frames = tracking_results[person_id]['frames']
            #print('22222222',bboxes)
            dataset = Inference(frame=frame_orig,bboxes=bboxes,scale=bbox_scale)

            with torch.no_grad():

                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                batch = dataset
                batch = batch.unsqueeze(0).unsqueeze(0)
                batch = batch.to(device)
                #print(batch.shape)
                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))


                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)

                del batch

            pred_cam = pred_cam.cpu().numpy()
            #print('pred_cam:  ',pred_cam)  #不同
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            #print('3333333333',pred_cam)
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )
            #print('orig_cam',orig_cam.shape)
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

        #print('vibe_results orig_cam:  ',vibe_results[1]['orig_cam'])
        #print('vibe_results pose:  ', vibe_results[1]['pose'])
        end = time.time()
        fps = 1 / (end - vibe_time)

        print(f'VIBE FPS: {fps:.2f}')


        if not args.no_render:
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

            #img = frame

            if args.sideview:
                side_img = np.zeros_like(img)

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

            #img = img.numpy()
            out.write(img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        total_time = time.time() - total_time
        i += 1
        time_acc += total_time
        print('num of frame: ',i,f'  Total time spent: {total_time:.2f} seconds (detect+track+vibe+render).')
        print(f'FPS : { 1 / total_time:.2f}.')
    print('Total average FPS: ',i / time_acc)
        # ========= Save rendered video ========= #
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='sample_video.mp4',
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,default='output/',
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking') #12

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE') #450

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()

    main(args)