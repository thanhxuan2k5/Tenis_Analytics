import cv2
from utils import (read_video,
                   save_video)
from tracker import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    #Read video input
    input_video_path = "input/input_video.mp4"
    video_frames = read_video(input_video_path)

    #Detect Players
    player_tracker = PlayerTracker(model_path='yolov8n')
    ball_tracker = BallTracker(model_path=r'D:/Tenis_Analysts/model/best.pt')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)


    #Court Line Detection model
    court_model_path =  r"D:/Tenis_Analysts/model/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #choosen players
    player_detections = player_tracker.choose_and_filter_player(court_keypoints,player_detections)
    #Draw output

    #Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_boxes(video_frames, ball_detections)
    #Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoint_on_video(output_video_frames, court_keypoints)

    #Draw frame number on top left conner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    save_video(output_video_frames,"output_video/output.mp4")

if __name__ == '__main__':
    main()