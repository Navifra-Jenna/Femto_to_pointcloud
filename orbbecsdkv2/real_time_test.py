import os
import cv2
import numpy as np
import open3d as o3d
from pyorbbecsdk import *
from ultralytics import YOLO
from typing import Optional
import time # 시간 측정을 위해 추가

# ----------------------------------------------------------------------
# 1. 설정 및 경로 설정
# ----------------------------------------------------------------------
ROOT_DIR = './real_time_test'
INFER_DIR = os.path.join(ROOT_DIR, 'inferenced_realtime')
os.makedirs(INFER_DIR, exist_ok=True)

# 카메라 파라미터 (사용자 장치에 맞게 확인 필요)
INTRINSICS_FX, INTRINSICS_FY = 926.3, 926.3
INTRINSICS_CX, INTRINSICS_CY = 640.0, 360.0

# YOLO 모델 로드
try:
    YOLO_MODEL = YOLO('./runs/segment/train5/weights/best.pt')
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}")
    YOLO_MODEL = None

# ----------------------------------------------------------------------
# 2. Orbbec SDK 헬퍼 함수
# ----------------------------------------------------------------------
def frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:
    """VideoFrame을 BGR NumPy 배열로 변환"""
    width, height = frame.get_width(), frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        return data.reshape((height, width, 3))
    elif color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None

def process_depth_raw(frames: FrameSet) -> Optional[np.ndarray]:
    """FrameSet에서 Depth Raw Data를 추출"""
    depth_frame = frames.get_depth_frame()
    if not depth_frame: return None
    width, height = depth_frame.get_width(), depth_frame.get_height()
    return np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))

# ----------------------------------------------------------------------
# 3. 실시간 포인트클라우드 생성 함수
# ----------------------------------------------------------------------
def generate_pcd_from_memory(color_bgr, depth_raw, yolo_result):
    """메모리 상의 데이터를 기반으로 PCD와 BBox 생성"""
    h, w = depth_raw.shape  # Depth 해상도 (576x640 등)
    
    # --- [수정] 컬러 이미지를 Depth 해상도에 맞게 리사이즈 ---
    # IndexError를 방지하기 위해 color 이미지를 depth와 동일한 크기로 맞춥니다.
    color_bgr_resized = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    color_rgb = cv2.cvtColor(color_bgr_resized, cv2.COLOR_BGR2RGB)
    
    # YOLO 세그멘테이션 마스크 추출
    mask = np.zeros((h, w), dtype=bool)
    if yolo_result[0].masks is not None:
        for m in yolo_result[0].masks.data:
            m_np = m.cpu().numpy()
            # YOLO 마스크도 Depth 해상도(h, w)에 맞게 리사이즈
            m_resized = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST) 
            mask |= (m_resized > 0.5)

    # 3D 좌표 변환
    depth_m = depth_raw.astype(np.float64) / 1000.0
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_m
    
    X = (u - INTRINSICS_CX) * Z / INTRINSICS_FX
    Y = (v - INTRINSICS_CY) * Z / INTRINSICS_FY
    pts = np.stack((X, Y, Z), axis=-1)
    
    # 마스크 및 유효 거리 필터링
    final_mask = (Z > 0.2) & (Z < 5.0) & mask
    
    # 이제 color_rgb와 final_mask의 크기가 동일하므로 에러가 발생하지 않습니다.
    valid_pts = pts[final_mask]
    valid_clr = color_rgb[final_mask]
    
    if valid_pts.size == 0: return None, None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts)
    pcd.colors = o3d.utility.Vector3dVector(valid_clr / 255.0)
    
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0) 
    
    return pcd, bbox

# ----------------------------------------------------------------------
# 4. 메인 실시간 루프
# ----------------------------------------------------------------------
def main():
    if YOLO_MODEL is None:
        print("YOLO 모델 로드 오류로 프로그램을 종료합니다.")
        return
        
    # 카메라 설정
    config = Config()
    pipeline = Pipeline()
    
    try:
        # 스트림 활성화 및 정렬 설정
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        
        config.set_align_mode(OBAlignMode.SW_MODE) # D2C 정렬 필수
        pipeline.start(config)
    except Exception as e:
        print(f"카메라 시작 실패: {e}")
        return

    # Open3D 비주얼라이저 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-time Orbbec 3D Inference (FPS Monitored)", width=1280, height=720)
    
    pcd_added = False
    cnt = 0
    start_time = time.time()

    print("실시간 처리를 시작합니다. 'q'를 누르면 종료합니다.")
    print("--------------------------------------------------")
    
    try:
        while True:
            # --- 1. 프레임 획득 (처리 속도가 느리면 여기서 기다림) ---
            frames = pipeline.wait_for_frames(100) 
            if frames is None: continue
            
            # --- 2. 데이터 추출 ---
            color_frame = frames.get_color_frame()
            # process_depth_raw 함수는 FrameSet을 직접 받음
            depth_raw = process_depth_raw(frames) 
            if color_frame is None or depth_raw is None: continue
            
            # BGR 이미지 변환 (YOLO 입력 및 CV2 프리뷰용)
            color_bgr = frame_to_bgr_image(color_frame)
            if color_bgr is None: continue

            # --- 3. YOLO Inference (여기서 처리 시간이 가장 오래 걸림) ---
            # NOTE: imgsz를 줄이면 속도 향상 가능 (예: imgsz=640)
            results = YOLO_MODEL.predict(source=color_bgr, save=False, imgsz=1280, verbose=False)
            
            # --- 4. 3D 처리 및 객체 생성 ---
            pcd, bbox = generate_pcd_from_memory(color_bgr, depth_raw, results)

            # --- 5. 시각화 업데이트 ---
            vis.clear_geometries()
            if pcd:
                vis.add_geometry(pcd)
                vis.add_geometry(bbox)
                if not pcd_added:
                    vis.reset_view_point(True)
                    pcd_added = True
            
            vis.poll_events()
            vis.update_renderer()
            
            # --- 6. 2D 프리뷰 및 FPS 계산 ---
            
            # FPS 계산 및 출력
            cnt += 1
            elapsed_time = time.time() - start_time
            current_fps = cnt / elapsed_time
            
            # 텍스트 오버레이 추가
            cv2.putText(color_bgr, f"FPS: {current_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("2D Preview (Press 'q' to quit)", color_bgr)
            
            # 결과 저장 (선택 사항)
            if cnt % 30 == 0: 
                label_path = os.path.join(INFER_DIR, f"infer_{cnt:06d}.txt")
                results[0].save_txt(label_path)
                print(f"Saved inference result for frame {cnt}. Current FPS: {current_fps:.2f}")


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\n프로그램 종료.")
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()