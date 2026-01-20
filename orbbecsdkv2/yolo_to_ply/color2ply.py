import numpy as np
import open3d as o3d
import cv2
import os
import re

# ----------------------------------------------------------------------
# 1. 설정 및 경로 (기존과 동일)
# ----------------------------------------------------------------------
BASE_PATH = '/home/jenna/workspace/projects/orbbecsdkv2/output/RecordFile_revised'
DEPTH_DIR = 'depth_raw_npy'
COLOR_DIR = 'color_vis_png'
YOLO_DIR = '/home/jenna/workspace/projects/orbbecsdkv2/text_only_output_2nd/predict/labels' 

INTRINSICS_FX, INTRINSICS_FY = 1129.472290,1128.802979
INTRINSICS_CX, INTRINSICS_CY = 958.838257, 552.6593
#color frame intrinsic parameters.
#depth frames were aligned and resized based on color frame.

# ----------------------------------------------------------------------
# 2. 전역 상태 및 콜백
# ----------------------------------------------------------------------
g_state = {
    'current_index': 0,
    'action': None,  # 'NEXT', 'PREV', 'QUIT'
    'num_frames': 0,
    'matched_files': [],
    'is_running': True
}

def key_callback_next(vis):
    global g_state
    g_state['action'] = 'NEXT'

def key_callback_prev(vis):
    global g_state
    g_state['action'] = 'PREV'

def key_callback_quit(vis):
    global g_state
    g_state['action'] = 'QUIT'
    g_state['is_running'] = False
    vis.close()

# ----------------------------------------------------------------------
# 3. 데이터 로직 (기존과 동일)
# ----------------------------------------------------------------------
def get_matched_files(base_path, depth_dir, color_dir, yolo_dir):
    depth_files = os.listdir(os.path.join(base_path, depth_dir))
    color_files = os.listdir(os.path.join(base_path, color_dir))
    yolo_files = os.listdir(yolo_dir)
    index_pattern = re.compile(r'(\d+)')
    depth_map = {index_pattern.findall(f)[-1]: f for f in depth_files if index_pattern.search(f)}
    color_map = {index_pattern.findall(f)[-1]: f for f in color_files if index_pattern.search(f)}
    yolo_map = {index_pattern.findall(f)[-1]: f for f in yolo_files if index_pattern.search(f)}
    common_indices = sorted(list(depth_map.keys() & color_map.keys() & yolo_map.keys()))
    if not common_indices: raise FileNotFoundError("매칭 실패")
    return [{'depth_path': os.path.join(base_path, depth_dir, depth_map[idx]),
             'color_path': os.path.join(base_path, color_dir, color_map[idx]),
             'yolo_path': os.path.join(yolo_dir, yolo_map[idx]), 'index': idx} for idx in common_indices]



def load_and_generate(file_info):
    '''
    
    1. 단위 변환: 실세계 물리 계산을 위해 mm를 m로 바꿉니다.
    2. Intrinsics(내부 파라미터) 활용: 카메라 렌즈의 특성(초점거리, 중심점)을 이용해 2차원 점을 3차원으로 뿜어냅니다.
    3. YOLO 결합: 이미지 전체를 3D로 만들면 노이즈가 많으므로, AI(YOLO)가 찾은 물체 영역만 마스킹하여 정밀한 데이터만 남깁니다.
    4. 회전 보정: 카메라 좌표계와 시각화 좌표계의 차이(upside-down 뒤집힘)를 수학적으로 정렬합니다.

    '''
    # 1. 데이터 로드: 저장된 Numpy(Depth)와 이미지(Color) 파일을 읽어옵니다.
    depth_data = np.load(file_info['depth_path'])  # mm 단위의 깊이 정보 (2D 배열)
    color_bgr = cv2.imread(file_info['color_path']) # OpenCV는 기본적으로 BGR 순서로 읽음
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) # 3D 시각화를 위해 RGB로 변환
    
    h, w = depth_data.shape[:2] # 이미지의 세로(h), 가로(w) 크기 추출
    print('h : ', h, 'w : ',w)
    # 2. 마스크 생성: YOLO 세그멘테이션 정보를 읽어 3D로 만들 영역만 골라냅니다.
    mask = np.full((h, w), False, dtype=bool) # 처음엔 모든 픽셀을 제외(False)로 설정
    if os.path.exists(file_info['yolo_path']):
        with open(file_info['yolo_path'], 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                # YOLO 포맷: [class_id, x1, y1, x2, y2, ... confidence]
                # 정규화된 좌표(0~1)를 실제 픽셀 좌표(0~w, 0~h)로 복원
                norm_coords = parts[1:-1] 
                pixel_coords = [[int(norm_coords[i]*w), int(norm_coords[i+1]*h)] 
                                for i in range(0, len(norm_coords), 2)]
                
                # 다각형(Polygon) 내부를 1로 채운 임시 마스크 생성
                temp = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(temp, [np.array(pixel_coords, np.int32)], 1)
                # 비트 연산(OR)을 통해 여러 객체의 마스크를 하나로 합침
                mask |= temp.astype(bool)

    # 3. 좌표계 변환 준비: mm 단위를 m(미터)로 바꾸고 격자 지도를 만듭니다.
    depth_m = depth_data.astype(np.float64) / 1000.0 # mm -> m 단위 변환
    u, v = np.meshgrid(np.arange(w), np.arange(h)) # 모든 픽셀의 (u, v) 좌표 행렬 생성
    
    # 4. 역투영(Back-projection): 2D 픽셀을 3D 공간 좌표(X, Y, Z)로 계산
    # 수식: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy
    Z = depth_m
    X = (u - INTRINSICS_CX) * Z / INTRINSICS_FX
    Y = (v - INTRINSICS_CY) * Z / INTRINSICS_FY
    # (h, w, 3) 형태의 3D 좌표 지도 생성
    pts = np.stack((X, Y, Z), axis=-1)
    

    # 5. 유효 데이터 필터링: 거리 범위(0.2m~5m) 안의 데이터 + YOLO 마스크 영역만 추출
    final_mask = (Z > 0.2) & (Z < 5.0) & mask
    valid_pts = pts[final_mask]     # 필터링된 3D 좌표들
    valid_clr = color_rgb[final_mask] # 필터링된 좌표에 대응하는 색상들
    
    if valid_pts.size == 0: return None, None # 유효한 점이 없으면 종료

    # 6. Open3D 객체 생성: 리스트 형태의 데이터를 PointCloud 객체로 변환
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts)
    pcd.colors = o3d.utility.Vector3dVector(valid_clr / 255.0) # 0~1 사이 값으로 정규화

    # 7. 좌표계 보정 (가장 중요한 부분)
    # 이미지 좌표계(Y-down)는 아래쪽이 +Y이지만, 3D 시각화는 위쪽이 +Y입니다.
    # 
    # 따라서 X축을 회전축으로 하여 180도(pi) 회전시켜 위아래를 뒤집어줍니다.
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)) # 회전 행렬 생성
    pcd.rotate(R, center=(0, 0, 0)) # 원점 기준 회전 수행

    # 8. 경계 상자(Bounding Box) 생성: 추출된 점군을 감싸는 박스를 만듭니다.
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0) # 박스 색상을 빨간색으로 설정
    
    return pcd, bbox

# ----------------------------------------------------------------------
# 4. 메인 루프
# ----------------------------------------------------------------------
def main():
    '''
    g_state 전역 변수 활용:

    키보드 이벤트는 "별도의 스레드"처럼 동작하기 때문에, 메인 루프와 소통하기 위해 g_state라는 전역 딕셔너리를 사용합니다.

    사용자가 키를 누르면 콜백 함수가 g_state['action'] 값을 바꾸고, 메인 루프가 이 값을 확인하여 실제 작업을 수행하는 구조입니다.

    vis.clear_geometries():

    이 함수가 없으면 프레임을 넘길 때마다 이전 프레임의 점들이 겹쳐서 보입니다. 매번 깨끗이 지우고 새로 그리는 것이 핵심입니다.

    vis.poll_events() & vis.update_renderer():

    Open3D 뷰어가 멈춰있지 않고 마우스로 3D 모델을 돌려보거나 창을 옮기는 등의 동작을 가능하게 해주는 핵심 엔진입니다.
    '''
    global g_state
    # 1. 파일 매칭 및 초기화: Depth, Color, YOLO 파일들을 짝지어 리스트로 만듭니다.
    g_state['matched_files'] = get_matched_files(BASE_PATH, DEPTH_DIR, COLOR_DIR, YOLO_DIR)
    g_state['num_frames'] = len(g_state['matched_files'])
    
    # 특정 인덱스(000075)부터 시작하도록 설정 (없으면 0번부터)
    start_idx = '000075'
    try:
        g_state['current_index'] = next(i for i, f in enumerate(g_state['matched_files']) if f['index'] == start_idx)
    except StopIteration:
        g_state['current_index'] = 0

    # 2. 시각화 창(Window) 생성 및 설정
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Orbbec 3D Viewer (R: Next, L: Prev, Q: Quit)", width=1280, height=720)

    # 3. 키보드 콜백(Callback) 등록
    # 사용자가 특정 키를 누르면 미리 정의된 함수(key_callback_next 등)가 실행되도록 연결합니다.
    vis.register_key_callback(ord('R'), key_callback_next) # R키: 다음 프레임
    vis.register_key_callback(ord('L'), key_callback_prev) # L키: 이전 프레임
    vis.register_key_callback(ord('Q'), key_callback_quit) # Q키: 종료

    # 빈 객체 초기화 (데이터를 담을 그릇들)
    pcd_obj = o3d.geometry.PointCloud()
    bbox_obj = o3d.geometry.LineSet()
    
    first_load = True # 처음 실행 시 무조건 데이터를 로드하기 위한 플래그

    # 4. 무한 루프 시작: 사용자가 종료(Q)할 때까지 반복
    while g_state['is_running']:
        
        # [조건부 로드] 사용자가 키를 눌러 액션이 발생했거나, 첫 실행일 때만 실행
        if g_state['action'] is not None or first_load:
            
            # 인덱스 계산: 현재 몇 번째 파일을 보여줄지 결정 (순환 구조)
            if g_state['action'] == 'NEXT':
                g_state['current_index'] = (g_state['current_index'] + 1) % g_state['num_frames']
            elif g_state['action'] == 'PREV':
                g_state['current_index'] = (g_state['current_index'] - 1) % g_state['num_frames']
            
            g_state['action'] = None # 처리 완료 후 액션 초기화 (계속 넘어가 방지)
            
            # 5. 데이터 로드 및 3D 생성 (앞서 설명한 load_and_generate 함수 호출)
            file_info = g_state['matched_files'][g_state['current_index']]
            pcd, bbox = load_and_generate(file_info)
            
            # 6. 화면 갱신: 기존에 그려진 것들을 지우고 새로 로드된 데이터를 추가
            vis.clear_geometries()
            if pcd:
                vis.add_geometry(pcd)
                vis.add_geometry(bbox)
                
                # 첫 로드 시에만 카메라 시점을 데이터에 맞춰 초기화
                if first_load:
                    vis.reset_view_point(True)
                    first_load = False
            
            print(f"Frame: {file_info['index']} Loaded.")

        # 7. 시스템 이벤트 처리 (창 닫기, 마우스 드래그, 화면 업데이트 등)
        if not vis.poll_events(): # 창이 닫히거나 문제가 생기면 루프 탈출
            break
        vis.update_renderer() # 변경된 내용을 실제 그래픽 카드로 전송하여 화면에 그림

    # 8. 종료 처리: 창을 닫고 자원을 해제
    vis.destroy_window()

if __name__ == '__main__':
    main()