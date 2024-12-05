import cv2
import face_recognition
import os
import glob
from tqdm import tqdm
import time


# Function to process a single video and save frames to the output folder
def process_video(video_path, output_frames_dir, num_frames=30):
    # Create the processed_videos.txt file if it doesn't exist
    processed_videos_path = os.path.join(output_frames_dir, "metadata.txt")
    if not os.path.exists(processed_videos_path):
        open(processed_videos_path, 'a').close()

    # Check if the video has already been processed
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    with open(processed_videos_path, 'r') as file:
        processed_videos = file.read().splitlines()

    if video_name in processed_videos:
        print(f"Frames for {video_name} already extracted. Skipping...")
        return

    try:
        video_capture = cv2.VideoCapture(video_path)

        frame_count = 0
        all_frames = []

        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()

            if not ret:
                break
            frame_count += 1
            all_frames.append(frame)

        num_frames_per_section = num_frames // 3

        # Select the first 10 frames
        first_10_frames = all_frames[:num_frames_per_section]

        # Select the middle 10 frames
        middle_start = frame_count // 2 - (num_frames_per_section // 2)
        middle_end = middle_start + num_frames_per_section
        median_position = (frame_count + 1) // 2
        if middle_start <= median_position <= middle_end:
            middle_start = median_position - (num_frames_per_section // 2)
            middle_end = median_position + (num_frames_per_section // 2)

        # Select the middle 10 frames
        middle_10_frames = all_frames[middle_start:middle_end]

        # Select the last 10 frames
        last_10_frames = all_frames[-num_frames_per_section:]

        # Combine the selected frames
        selected_frames = first_10_frames + middle_10_frames + last_10_frames
        frame_counter = 0
        with tqdm(total=len(selected_frames), desc=f"Processing {video_name}") as pbar:
            for i, selected_frame in enumerate(selected_frames):
                face_locations = face_recognition.face_locations(selected_frame)

                for j, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location

                    face_image = selected_frame[top:bottom, left:right]

                    output_face_path = os.path.join(output_frames_dir, f"{video_name}_frame_{i}_face_{j}.jpg")
                    cv2.imwrite(output_face_path, face_image)
                    frame_counter += 1
                pbar.update(1)  # update the progress bar

        with open(processed_videos_path, 'a') as file:
            file.write(video_name + '\n')
        video_capture.release()

        print(f"\n{frame_counter} frames with faces extracted from {video_name} and saved to {output_frames_dir}\n")

    except KeyboardInterrupt:
        raise KeyboardInterrupt(f"Processing interrupted by user for {video_name}.")
    except RuntimeError as e:
        raise RuntimeError(f"RuntimeError occurred for {video_name}. Error: {str(e)}")
    except SystemExit:
        raise SystemExit(f"SystemExit occurred for {video_name}.")
    except InterruptedError:
        raise InterruptedError(f"InterruptedError occurred for {video_name}.")
    except MemoryError:
        raise MemoryError(f"MemoryError occurred for {video_name}.")
    except OSError as e:
        raise OSError(f"OSError occurred for {video_name}. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred for {video_name}. Error: {str(e)}")


video_folder = '/content/drive/MyDrive/checking/'
output_frames_dir = '/content/drive/MyDrive/check-dataset/'
video_paths = glob.glob(os.path.join(video_folder, '*.mp4'))

# Process each video in the list
start_time = time.time()
for video_path in video_paths:
    process_video(video_path, output_frames_dir, num_frames=30)
    # print()

end_time = time.time()
print(f"\n\n\nProcessing completed successfully in {round(end_time - start_time, 3)} seconds.")
