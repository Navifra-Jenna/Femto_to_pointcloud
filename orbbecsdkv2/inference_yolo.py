from ultralytics import YOLO

# 모델 로드
model = YOLO('./runs/segment/train5/weights/best.pt') 

# 추론 실행 (save=False로 시각화된 이미지 저장을 명시적으로 비활성화)
model.predict(
    source='/home/jenna/workspace/projects/orbbecsdkv2/output/RecordFile_revised/color_vis_png', # Directory to inference
    save=False,          # <--- 중요: 시각화 이미지 저장을 막음
    save_txt=True,       # 라벨 파일(.txt)만 저장 요청
    project='text_only_output_2nd', 
    save_conf = True, 
    imgsz = 1280

)