import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
import glob

# --- 설정 변수 ---
# 원본 데이터 경로
ORIGINAL_IMAGE_DIR = './output/RecordFile/color'
ORIGINAL_LABEL_DIR = './output/RecordFile/labels'

# 학습 데이터셋을 구성할 타겟 경로
TARGET_BASE_DIR = './yolo_train_dataset'

# 분할 비율
TRAIN_RATIO = 0.8  # 학습 데이터 비율
VAL_RATIO = 0.2    # 검증 데이터 비율

# YOLOv11 dataset.yaml 설정 내용
YOLO_CONFIG = {
    'path': f'..',            # dataset.yaml의 위치(A)를 기준으로 상위 폴더를 가리킴
    'train': 'yolo_train_dataset/images/train',
    'val': 'yolo_train_dataset/images/val',
    'nc': 1,
    'names': ['palette_0']
}

def split_and_setup_yolo_dataset():
    # 1. 경로 존재 여부 확인
    if not os.path.exists(ORIGINAL_IMAGE_DIR) or not os.path.exists(ORIGINAL_LABEL_DIR):
        print(f"⚠️ 에러: 원본 이미지 폴더({ORIGINAL_IMAGE_DIR}) 또는 라벨 폴더({ORIGINAL_LABEL_DIR})를 찾을 수 없습니다.")
        return

    # 2. 타겟 디렉토리 구조 생성
    target_img_train = os.path.join(TARGET_BASE_DIR, 'images', 'train')
    target_img_val = os.path.join(TARGET_BASE_DIR, 'images', 'val')
    target_lbl_train = os.path.join(TARGET_BASE_DIR, 'labels', 'train')
    target_lbl_val = os.path.join(TARGET_BASE_DIR, 'labels', 'val')

    # 필요한 모든 폴더 생성
    os.makedirs(target_img_train, exist_ok=True)
    os.makedirs(target_img_val, exist_ok=True)
    os.makedirs(target_lbl_train, exist_ok=True)
    os.makedirs(target_lbl_val, exist_ok=True)

    # 3. 데이터 목록 수집
    # 이미지 파일 목록을 가져옵니다. (.jpg, .png 등 모든 일반적인 이미지 확장자)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(ORIGINAL_IMAGE_DIR, ext)))

    # 이미지 파일의 '기본 이름'(확장자 제외) 목록을 만듭니다. 이것이 데이터셋의 고유 ID가 됩니다.
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    
    if not base_names:
        print(f"⚠️ 경고: {ORIGINAL_IMAGE_DIR} 폴더에서 이미지 파일을 찾을 수 없습니다.")
        return

    # 4. 학습/검증 데이터 분할
    train_names, val_names = train_test_split(
        base_names,
        test_size=VAL_RATIO,
        random_state=42  # 재현성을 위한 시드값
    )

    print(f"\n총 데이터 개수: {len(base_names)}개")
    print(f"-> 학습 데이터: {len(train_names)}개 (비율: {TRAIN_RATIO})")
    print(f"-> 검증 데이터: {len(val_names)}개 (비율: {VAL_RATIO})")

    # 5. 분할된 파일 복사 및 이동
    def copy_files(name_list, img_dest, lbl_dest):
        copied_count = 0
        for base_name in name_list:
            # 원본 이미지 파일 찾기 (확장자 고려)
            img_src_path_list = glob.glob(os.path.join(ORIGINAL_IMAGE_DIR, f"{base_name}.*"))
            if not img_src_path_list:
                # 이미지가 없으면 건너뜀
                continue
            img_src_path = img_src_path_list[0] # 첫 번째로 찾은 이미지 파일을 사용

            # 원본 레이블 파일 경로 (YOLO segmentation은 일반적으로 .txt 파일)
            lbl_src_path = os.path.join(ORIGINAL_LABEL_DIR, f"{base_name}.txt") 

            # 이미지 복사
            shutil.copy(img_src_path, os.path.join(img_dest, os.path.basename(img_src_path)))
            
            # 레이블 파일 복사
            if os.path.exists(lbl_src_path):
                shutil.copy(lbl_src_path, os.path.join(lbl_dest, os.path.basename(lbl_src_path)))
                copied_count += 1
            else:
                print(f"레이블 파일 누락: {base_name}.txt. 해당 이미지는 데이터셋에서 제외됩니다.")
            
        return copied_count

    print("\n📦 학습 데이터 복사 중...")
    copy_files(train_names, target_img_train, target_lbl_train)
    print("📦 검증 데이터 복사 중...")
    copy_files(val_names, target_img_val, target_lbl_val)
    
    # 6. YAML 파일 생성
    yaml_path = os.path.join(TARGET_BASE_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(YOLO_CONFIG, f, default_flow_style=False)
    
    print(f"\n✅ 'dataset.yaml' 파일이 생성되었습니다: {yaml_path}")
    print(yaml.dump(YOLO_CONFIG, default_flow_style=False))
    
    print("\n🎉 데이터셋 구성 및 분할이 완료되었습니다!")
    print(f"이제 '{TARGET_BASE_DIR}' 폴더를 학습 경로로 사용할 수 있습니다.")

# --- 실행 ---
# 데이터를 분할하기 전에 'scikit-learn'이 설치되어 있어야 합니다: pip install scikit-learn
# 그리고 'B/color'와 'B/labels' 디렉토리가 존재하고 내부에 파일이 있어야 합니다.
split_and_setup_yolo_dataset()