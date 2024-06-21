import os
import cv2

 

def extract_frames_from_videos(video_list, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

 

    for video_file in video_list:
        video_path = os.path.join(output_folder, video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        frame_output_folder = os.path.join(output_folder, video_name)

        # Create a subfolder for each video to store frames
        if not os.path.exists(frame_output_folder):
            os.makedirs(frame_output_folder)

 

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

 

        for frame_number in range(frame_count):
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break

 

            # Save the frame as an image
            frame_filename = f"{video_name}_frame_{frame_number:04d}.jpg"
            frame_filepath = os.path.join(frame_output_folder, frame_filename)
            cv2.imwrite(frame_filepath, frame)

 

        # Release the video capture object
        cap.release()

 

if __name__ == "__main__":
    video_list = ["D:/Saikrishna/Downloads/videosppe/958f8a00-34ee-488b-bc96-c7dc4863a611.mp4", "D:/Saikrishna/Downloads/videosppe/5512e156-5438-4e23-9846-899615755a92.mp4","D:/Saikrishna/Downloads/videosppe/a31579ff-a107-4295-a222-98ca8d9d6662.mp4"]  # Add your video filenames here
    output_folder = "frames_output"  # Replace with the desired output folder name
    extract_frames_from_videos(video_list, output_folder)

