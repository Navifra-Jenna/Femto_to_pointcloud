# ******************************************************************************
#  Copyright (c) 2023-2024 Orbbec 3D Technology, Inc
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
from typing import Union, Any, Optional

# --- Color Frame 변환 헬퍼 함수 ---
# pyorbbecsdk에서 지원하지 않는 일부 YUV/YCBCR 포맷을 BGR로 변환하는 함수들입니다.
# 현재 코드에서는 frame_to_bgr_image 함수를 통해 내부적으로 사용됩니다.

def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """YUYV 포맷을 BGR 포맷으로 변환"""
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """UYVY 포맷을 BGR 포맷으로 변환"""
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """I420 포맷을 BGR 포맷으로 변환"""
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """NV21 포맷을 BGR 포맷으로 변환"""
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """NV12 포맷을 BGR 포맷으로 변환"""
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def determine_convert_format(frame: VideoFrame):
    """프레임 포맷에 따라 pyorbbecsdk 변환 포맷 결정"""
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
    """VideoFrame을 pyorbbecsdk 필터를 사용하여 RGB VideoFrame으로 변환"""
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print("Convert {} to RGB failed".format(frame.get_format()))
    return rgb_frame


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    """VideoFrame을 OpenCV 이미지 (BGR numpy array)로 변환"""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    # image = np.zeros((height, width, 3), dtype=np.uint8) # 이 라인은 불필요하여 제거 가능하나, 기능 보존을 위해 유지

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환은 시각화에 적합한지는 확인 필요 (대부분 BGR로 처리)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


ESC_KEY = 27
is_paused = False
# 캐시된 프레임 (현재 코드에서는 사용되지 않음)
cached_frames = {
    'color': None,
    'depth': None,
}

# (setup_camera 함수는 사용되지 않으므로 주석 처리하거나 제거 가능하나, 원본 유지)
# def setup_camera():
#     """Setup camera and stream configuration"""
#     pipeline = Pipeline()
#     config = Config()
#     device = pipeline.get_device()

#     # Try to enable all possible sensors
#     video_sensors = [
#         OBSensorType.COLOR_SENSOR,
#         OBSensorType.DEPTH_SENSOR,
#     ]
#     sensor_list = device.get_sensor_list()
#     for sensor in range(len(sensor_list)):
#         try:
#             sensor_type = sensor_list[sensor].get_type()
#             if sensor_type in video_sensors:
#                 config.enable_stream(sensor_type)
#         except:
#             continue

#     pipeline.start(config)
#     return pipeline

def process_color(frame):
    """프레임셋에서 Color VideoFrame을 추출하고 BGR 이미지로 변환"""
    if not frame:
        return None
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None
    try:
        # frame_to_bgr_image 함수는 BGR NumPy 배열을 반환합니다.
        return frame_to_bgr_image(color_frame) 
    except ValueError:
        print("Error processing color frame")
        return None
        
def process_depth(frame, scale=1):
    """프레임셋에서 Depth VideoFrame을 추출하고 Raw Depth Data (np.uint16)로 변환"""
    if not frame:
        return None
    depth_frame = frame.get_depth_frame()
    if not depth_frame:
        return None
    try:         
        width, height = depth_frame.get_width(), depth_frame.get_height()
        # Raw Depth Data (uint16)를 NumPy 배열로 변환
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_data = depth_raw * scale
        return depth_data
    except ValueError:
        print("Error processing depth frame")
        return None

# (create_display 함수는 현재 코드에서 사용되지 않으므로 주석 처리하거나 제거 가능하나, 원본 유지)
# def create_display(frames, width=1280, height=720):
#     """Create display window"""
#     display = np.zeros((height, width, 3), dtype=np.uint8)
#     h, w = height // 2, width // 2

#     # Process video frames
#     if 'color' in frames and frames['color'] is not None:
#         display[0:h, 0:w] = cv2.resize(frames['color'], (w, h))

#     if 'depth' in frames and frames['depth'] is not None:
#         display[0:h, w:] = cv2.resize(frames['depth'], (w, h))

#     return display


def main():
    # --- 1. 디스플레이 초기 설정 ---
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

    # --- 2. 저장 경로 및 디렉토리 설정 ---
    base_save_path = './output/RecordFile_revised_2'
    
    # 시각화된 이미지 (.png) 저장 경로
    color_vis_dir = os.path.join(base_save_path, 'color_vis_png')
    depth_vis_dir = os.path.join(base_save_path, 'depth_vis_png')
    # Raw Depth Data (.npy) 저장 경로
    depth_raw_npy_dir = os.path.join(base_save_path, 'depth_raw_npy')

    os.makedirs(color_vis_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)
    os.makedirs(depth_raw_npy_dir, exist_ok=True)

    # --- 3. 카메라 및 스트림 설정 ---
    config = Config()
    config.set_align_mode(OBAlignMode.SW_MODE)
    pipeline = Pipeline()
    device = pipeline.get_device()
    # 장치 타이머를 호스트와 동기화
    device.timer_sync_with_host()

    try:
        # Depth 스트림 활성화
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if profile_list is None:
            print("No depth sensor found")
            return
        profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(profile)
    except Exception as e:
        print(e)
        return
        
    try:
        # Color 스트림 활성화 및 Depth와 정렬 설정
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            print("No color sensor found")
            return
        profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(profile)
        config.set_align_mode(OBAlignMode.SW_MODE)
    except Exception as e:
        print(e)
        
    # Color 스트림 기준으로 정렬하는 필터
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    pipeline.enable_frame_sync() # 프레임 동기화 활성화
    pipeline.start(config)
    
    cnt = 0
    global is_paused
    
    # --- 4. 메인 루프: 프레임 획득, 처리, 표시 및 저장 ---
    while True:
        try:
            # 프레임 획득
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue 
                
            # 프레임 정렬
            frames = align_filter.process(frames)
            if frames is None:
                continue 
            frames = frames.as_frame_set()
            if frames is None:
                continue 

            # 프레임 처리
            color_frame = process_color(frames) # BGR NumPy 배열 (Color Raw/Vis)
            depth_frame = process_depth(frames)  # uint16 NumPy 배열 (Depth Raw)
            
            # Depth 시각화 처리 (컬러맵 및 Color와의 혼합)
            depth_frame_visualize = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_frame_visualize = cv2.applyColorMap(depth_frame_visualize, cv2.COLORMAP_JET)
            depth_frame_visualize[depth_frame_visualize == 128] = 0
            # Color 이미지와 Depth 컬러맵 이미지를 50% 혼합하여 최종 시각화
            depth_frame_visualize = cv2.addWeighted(color_frame, 1, depth_frame_visualize, 0.5, 0)
            
            # 화면 표시
            cv2.imshow("Record Viewer", cv2.resize(depth_frame_visualize, resize))
            key = cv2.waitKey(1) & 0xFF
   
            # 파일 이름 정의
            frame_index = f'{cnt:06d}'
            color_name_png = f'color_frame_{frame_index}.png'
            depth_name_png = f'depth_frame_{frame_index}.png'
            depth_name_npy = f'depth_raw_frame_{frame_index}.npy'

            # 시각화된 프레임 저장 (.png)
            cv2.imwrite(os.path.join(color_vis_dir, color_name_png), color_frame)
            cv2.imwrite(os.path.join(depth_vis_dir, depth_name_png), depth_frame_visualize)

            # Raw Depth Data 저장 (.npy)
            np.save(os.path.join(depth_raw_npy_dir, depth_name_npy), depth_frame) 
            
            cnt += 1
            
            # 's' 키를 이용한 일시정지/재개 로직 (주석 처리됨, 기능 유지)
            # if key == ord('s'):
            #     if not is_paused:
            #         # recorder.pause()
            #         is_paused = True
            #         print("[PAUSED] Recording paused")
            #     else:
            #         # recorder.resume()
            #         is_paused = False
            #         print("[RESUMED] Recording resumed")
                    
            # 종료 조건 (q 또는 ESC)
            if key in (ord('q'), ESC_KEY):
                break
                
        except Exception as e:
            print(e)
            break
            
    # --- 5. 종료 처리 ---
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()