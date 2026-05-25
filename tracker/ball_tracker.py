from ultralytics import YOLO
import cv2
import pickle
import os
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self,ball_position):
        ball_position = [x.get(1,[]) for x in ball_position]

        #Convert the list from pandas to dataframe
        df_ball_position = pd.DataFrame(ball_position, columns=['x1','y1','x2','y2'])
        #interpolate the missing values
        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill()




    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            if os.path.exists(stub_path):
                with open(stub_path, 'rb') as f:
                    ball_detections = pickle.load(f)
                print(f"Đã đọc ball_detections từ {stub_path}")
                return ball_detections
            else:
                print(f"Cảnh báo: Tệp stub không tồn tại tại {stub_path}. Tiến hành phát hiện.")

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
            print(f"Đã lưu ball_detections vào {stub_path}")
            
        return ball_detections

    def detect_frame(self,frame):
        result = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        for box in result.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict
    def draw_boxes(self,video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items(): #Draw Bbox
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball Id: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            output_video_frames.append(frame)
        return output_video_frames
