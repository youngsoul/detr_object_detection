from opencv_utils import process_video

if __name__ == '__main__':
    video_path = "./images/my_videos/coyote_backyard.mp4"
    process_video(video_path, skip_first_frames=200, max_frames=100, threshold=0.7)
