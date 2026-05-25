from utils import (read_video,
                   save_video)
from tracker import PlayerTracker
def main():
    #Read video input
    input_video_path = "input/input_video.mp4"
    video_frames = read_video(input_video_path)

    #Detect Players
    player_tracker = PlayerTracker(model_path='yolov8n')
    player_detections = player_tracker.detect_frame(video_frames)

    #Draw output

    #Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detections)

    #Save Video
    save_video(output_video_frames,"output_video/output.mp4")

if __name__ == '__main__':
    main()