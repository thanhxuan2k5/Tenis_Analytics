from ultralytics import YOLO
import cv2
import pickle
import os

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            if os.path.exists(stub_path):
                with open(stub_path, 'rb') as f:
                    player_detections = pickle.load(f)
                print(f"Đã đọc player_detections từ {stub_path}")
                return player_detections
            else:
                print(f"Cảnh báo: Tệp stub không tồn tại tại {stub_path}. Tiến hành phát hiện.")

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            print(f"Đã lưu player_detections vào {stub_path}")
            
        return player_detections

    def detect_frame(self,frame):
        result = self.model.track(frame, persist=True)[0]
        id_name_dict = result.names

        player_dict = {}
        for box in result.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        return player_dict
    def draw_boxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items(): #Draw Bbox
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Player Id: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
            output_video_frames.append(frame)
        return output_video_frames
