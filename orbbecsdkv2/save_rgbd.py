# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import os
import cv2
import numpy as np
import screeninfo
from pyorbbecsdk import *
from datetime import datetime
from typing import Union, Any, Optional
# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************


def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def determine_convert_format(frame: VideoFrame):
    if frame.get_format() == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    elif frame.get_format() == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    elif frame.get_format() == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    elif frame.get_format() == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    elif frame.get_format() == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    elif frame.get_format() == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    else:
        return None


def frame_to_rgb_frame(frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    print("covert format: {}".format(convert_format))
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print("Convert {} to RGB failed".format(frame.get_format()))
    return rgb_frame


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


ESC_KEY = 27

is_paused = False
# cached frames for better visualization
cached_frames = {
    'color': None,
    'depth': None,
}

def setup_camera():
    """Setup camera and stream configuration"""
    pipeline = Pipeline()
    config = Config()
    device = pipeline.get_device()

    # Try to enable all possible sensors
    video_sensors = [
        OBSensorType.COLOR_SENSOR,
        OBSensorType.DEPTH_SENSOR,
    ]
    sensor_list = device.get_sensor_list()
    for sensor in range(len(sensor_list)):
        try:
            sensor_type = sensor_list[sensor].get_type()
            if sensor_type in video_sensors:
                config.enable_stream(sensor_type)
        except:
            continue

    pipeline.start(config)
    return pipeline

def process_color(frame):
    """Process color frame to BGR image"""
    if not frame:
        return None
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None
    try:
        return frame_to_bgr_image(color_frame)
    except ValueError:
        print("Error processing color frame")
        return None
        
def process_depth(frame, scale=1):
    """Process depth frame to colorized depth image"""
    if not frame:
        return None
    depth_frame = frame.get_depth_frame()
    if not depth_frame:
        return None
    try:         
        width, height = depth_frame.get_width(), depth_frame.get_height()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_data = depth_raw * scale
        return depth_data
    except ValueError:
        print("Error processing depth frame")
        return None

def create_display(frames, width=1280, height=720):
    """Create display window"""
    display = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = height // 2, width // 2

    # Process video frames
    if 'color' in frames and frames['color'] is not None:
        display[0:h, 0:w] = cv2.resize(frames['color'], (w, h))

    if 'depth' in frames and frames['depth'] is not None:
        display[0:h, w:] = cv2.resize(frames['depth'], (w, h))

    return display


def main():

    screen = screeninfo.get_monitors()[0]
    sw, sh = screen.width, screen.height
    sratio = sw/sh
    size = (1472, 828)
    if sratio < (size[0]/size[1]):
        ratio = sw/size[0]
    else:
        ratio = sh/size[1]
    resize = (int(size[0]*ratio), int(size[1]*ratio))
    cv2.namedWindow('Record Viewer', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Record Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    save_path = './output/RecordFile'
    # os.makedirs(save_path, exist_ok=True)
    color_dir = os.path.join(save_path, 'color')
    depth_dir = os.path.join(save_path, 'depth')
    depth_raw_dir = os.path.join(save_path, 'depth_raw_npy') # Depth Raw (.npy)

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    +os.makedirs(depth_raw_dir, exist_ok=True)



    config = Config()
    config.set_align_mode(OBAlignMode.SW_MODE)
    pipeline = Pipeline()
    device = pipeline.get_device()
    #synchronize the timer of the device with the host
    device.timer_sync_with_host()
    # initialize recording

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if profile_list is None:
            print("No depth sensor found")
            return
        profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(profile)
    except Exception as e:
        print(e)
        return
    # enable color stream
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            print("No color sensor found")
            return
        profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(profile)
        config.set_align_mode(OBAlignMode.SW_MODE)
    except Exception as e:
        print(e)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


    pipeline.enable_frame_sync()
    pipeline.start(config)
    # recorder = RecordDevice(device, save_file)
    cnt = 0

    global is_paused
    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue 
            frames = align_filter.process(frames)
            if frames is None:
                continue 
            frames = frames.as_frame_set()
            if frames is None:
                continue 

            color_frame = process_color(frames)
            depth_frame = process_depth(frames)      
            
            depth_frame_visualize = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_frame_visualize = cv2.applyColorMap(depth_frame_visualize, cv2.COLORMAP_JET)
            depth_frame_visualize[depth_frame_visualize == 128] = 0
            depth_frame_visualize = cv2.addWeighted(color_frame, 1, depth_frame_visualize, 0.5, 0)
            cv2.imshow("Record Viewer", cv2.resize(depth_frame_visualize, resize))
            key = cv2.waitKey(1) & 0xFF
   
            color_name = f'color_frame_{cnt:06d}.png'
            depth_name = f'depth_frame_{cnt:06d}.png'

            depth_name_npy = f'depth_raw_frame_{cnt:06d}.npy'

            cv2.imwrite(os.path.join(color_dir, color_name), color_frame)
            cv2.imwrite(os.path.join(depth_dir, depth_name), depth_frame_visualize)

            np.save(os.path.join(depth_raw_dir, depth_name_npy), depth_frame) #saving depth raw value
            cnt += 1
            # if key == ord('s'):
            #     if not is_paused:0
            #         # recorder.pause()
            #         is_paused = True
            #         print("[PAUSED] Recording paused")
            #     else:
            #         # recorder.resume()
            #         is_paused = False
            #         print("[RESUMED] Recording resumed")
            if key in (ord('q'), 27):
                break
        except Exception as e:
            print(e)
            break
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()