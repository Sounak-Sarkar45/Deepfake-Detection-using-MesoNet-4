import cv2
import face_recognition


# Load the video file
video_path = 'C://Users//sayan//OneDrive - vit.ac.in//VIT Study material//Third Semester//SET3//Data-SET//Celeb-df//Celeb-synthesis//id0_id2_0005.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a variable to keep track of the frame count
frame_count = 0

# Loop through the frames of the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame count
    frame_count += 1

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)

    # If faces are found in the frame, extract and save them
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Crop the face region from the frame
            face_image = frame[top:bottom, left:right]

            # Save or process the face_image as needed
            # For example, you can save it to a file:
            cv2.imwrite(f'face_{frame_count}.jpg', face_image)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
