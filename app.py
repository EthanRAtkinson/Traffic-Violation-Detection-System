import cv2
import numpy as np
from collections import deque
from collections import Counter
import os
import glob
import subprocess
import requests
import time
import shutil

#---------------OPEN CV Motion & Stop Detection------------------------------

# Define the path to the folder containing input clips and the base output folder for evidence.
video_folder = os.path.join("Footage")
evidence_folder = os.path.join("Evidence")
os.makedirs(evidence_folder, exist_ok=True)

# Get a list of all .mp4 files in the folder (adjust extension if needed)
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]

if not video_files:
    print("No video files found in", video_folder)
    exit(1)

# Process each video file in the folder
for video_filename in video_files:
    video_path = os.path.join(video_folder, video_filename)
    print("Processing video:", video_path)
    
    # Create a unique output folder for this video.
    # For example, if the video is "video1.mp4", the clips will be saved under "Evidence/video1/Clips".
    base_name = os.path.splitext(video_filename)[0]
    video_output_folder = os.path.join(evidence_folder, base_name, "Clips")
    os.makedirs(video_output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS and frame size from video.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    ret, sample_frame = cap.read()
    if not ret:
        print("Error reading the first frame from", video_filename)
        continue
    frame_height, frame_width = sample_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Background subtractor using MOG2
    backSub = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=250, detectShadows=True)

    # Warm-up frames to avoid detection boxes in the first few frames.
    WARMUP_FRAMES = 5  
    for i in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        backSub.apply(frame)
    
    # Smoothing for the bounding box: initialize as None.
    # This variable will hold the smoothed bounding box as (x, y, w, h)
    smoothed_box = None
    smoothing_factor = 0.15  # Lower values produce slower updates.
    
    # Count frames with no detection to allow for brief dropouts.
    lost_frames = 0
    lost_threshold = 10  # Number of consecutive frames with no detection before clearing the box.
    
    # Buffer to store recent centroid positions for stop detection.
    centroid_buffer = deque(maxlen=10)
    
    # Parameters for detecting if the vehicle has stopped.
    STOP_THRESHOLD = 3         # Maximum average movement (in pixels) allowed.
    stopped_frames = 0         # Counter for consecutive frames with low movement.
    required_stop_frames = 15  # Number of consecutive frames needed to mark "stopped".
    
    # Define area thresholds for valid contours.
    MIN_CONTOUR_AREA = 60000   # Filter out noise: too-small contours.
    MAX_CONTOUR_AREA = 150000  # Maximum allowed contour area for a normal update.
    
    # Variables for video clip recording.
    video_writer = None
    clip_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optionally, resize the frame (uncomment if desired)
        # frame = cv2.resize(frame, (640, 480))
        
        # Apply background subtraction to get the foreground mask.
        fgMask = backSub.apply(frame)
        
        # Clean up the mask with morphological operations.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgMask = cv2.dilate(fgMask, kernel, iterations=5)
        
        # Find contours in the foreground mask.
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detection_updated = False  # Flag to indicate a valid detection was found.

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_CONTOUR_AREA:
                lost_frames = 0  # Reset lost frame counter since we have a detection.
                detection_updated = True
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                if area < MAX_CONTOUR_AREA:
                    # Use the detected bounding box normally.
                    detected_box = (x, y, w, h)
                else:
                    # If detected area is too large, update the position but maintain clamped box size.
                    if smoothed_box is not None:
                        fixed_w, fixed_h = smoothed_box[2], smoothed_box[3]
                    else:
                        fixed_side = int(np.sqrt(MAX_CONTOUR_AREA))
                        fixed_w, fixed_h = fixed_side, fixed_side
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    new_x = int(center_x - fixed_w / 2)
                    new_y = int(center_y - fixed_h / 2)
                    detected_box = (new_x, new_y, fixed_w, fixed_h)
                
                # Update the smoothed box using exponential smoothing.
                if smoothed_box is None:
                    smoothed_box = detected_box
                else:
                    sx, sy, sw, sh = smoothed_box
                    nx, ny, nw, nh = detected_box
                    sx = int(smoothing_factor * nx + (1 - smoothing_factor) * sx)
                    sy = int(smoothing_factor * ny + (1 - smoothing_factor) * sy)
                    sw = int(smoothing_factor * nw + (1 - smoothing_factor) * sw)
                    sh = int(smoothing_factor * nh + (1 - smoothing_factor) * sh)
                    smoothed_box = (sx, sy, sw, sh)
        
        if not detection_updated:
            # Increment lost frame counter if no valid detection.
            lost_frames += 1
            if lost_frames >= lost_threshold:
                smoothed_box = None
                centroid_buffer.clear()
                stopped_frames = 0
        
        # If a smoothed bounding box exists, track and display it.
        if smoothed_box is not None:
            sx, sy, sw, sh = smoothed_box
            centroid = (int(sx + sw / 2), int(sy + sh / 2))
            centroid_buffer.append(centroid)
            
            # Draw the green (smoothed) bounding box and red centroid on the frame.
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            
            # Calculate displacement if we have enough history.
            if len(centroid_buffer) > 1:
                displacements = []
                pts = list(centroid_buffer)
                for pt1, pt2 in zip(pts, pts[1:]):
                    d = np.linalg.norm(np.array(pt2) - np.array(pt1))
                    displacements.append(d)
                avg_disp = sum(displacements) / len(displacements)
                
                if avg_disp < STOP_THRESHOLD:
                    stopped_frames += 1
                else:
                    stopped_frames = 0
            else:
                stopped_frames = 0
        else:
            centroid_buffer.clear()
            stopped_frames = 0
        
        # Display stop or moving status on the frame.
        if stopped_frames >= required_stop_frames:
            cv2.putText(frame, "Vehicle Stopped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Vehicle Moving", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        
        # Show the frame and foreground mask.
        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground Mask", fgMask)
        
        # ----- Video Recording Logic -----
        if smoothed_box is not None:
            # Start recording if detection exists and no writer is active.
            if video_writer is None:
                clip_index += 1
                clip_filename = f"{base_name}_clip_{clip_index}.mp4"
                clip_filepath = os.path.join(video_output_folder, clip_filename)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(clip_filepath, fourcc, fps, (frame_width, frame_height))
                print("Started recording clip:", clip_filepath)
            # Write the current frame (with detection overlay) to the clip.
            video_writer.write(frame)
        else:
            # If detection is lost and recording was active, release the writer to finalize the clip.
            if video_writer is not None:
                print("Stopped recording clip.")
                video_writer.release()
                video_writer = None

        
        # Exit video playback if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    cv2.destroyAllWindows()
#-----------------------------FFMpeg Frame Splitter-----------------------

# Set parameters for frame extraction and zooming
frame_rate = 2          # Number of frames per second
zoom_factor = 1.7       # e.g., 1.0 = no zoom, 1.7 = 70% zoom-in, etc.

# Define the parent directory
evidence_dir = 'Evidence'

# Loop through each item in the evidence folder
for subfolder in os.listdir(evidence_dir):
    subfolder_path = os.path.join(evidence_dir, subfolder)
    # Process only directories
    if os.path.isdir(subfolder_path):
        # Look for a "Clips" folder inside the subfolder
        clips_folder = os.path.join(subfolder_path, 'Clips')
        if os.path.isdir(clips_folder):
            # Create a "Frames" folder inside the current subfolder to save outputs
            frames_folder = os.path.join(subfolder_path, 'Frames')
            os.makedirs(frames_folder, exist_ok=True)
            
            # Find all mp4 files inside the current clips folder
            video_files = glob.glob(os.path.join(clips_folder, '*.mp4'))
            
            for video_file in video_files:
                # Extract the basename of the video file (without extension)
                video_basename = os.path.splitext(os.path.basename(video_file))[0]
                
                # Define the pattern for the output frames (e.g., video1_frame_0001.png)
                output_pattern = os.path.join(frames_folder, f'{video_basename}_frame_%04d.png')
                
                # Build the ffmpeg filter string to extract frames and apply zoom:
                # 1. fps: extract a given number of frames each second
                # 2. scale: enlarge the image according to the zoom factor
                # 3. crop: crop the center portion to restore the original size (zoom effect)
                vf_filter = (
                    f"fps={frame_rate},"
                    f"scale=iw*{zoom_factor}:ih*{zoom_factor},"
                    f"crop=iw/{zoom_factor}:ih/{zoom_factor}:" 
                    f"(iw-iw/{zoom_factor})/2:(ih-ih/{zoom_factor})/2"
                )
                
                # Build the ffmpeg command
                command = [
                    'ffmpeg',
                    '-i', video_file,
                    '-vf', vf_filter,
                    output_pattern
                ]
                
                # Run the ffmpeg command
                subprocess.run(command)

#------------Plate Recognizer--------------

def get_unique_folder_name(base_folder):
    """
    Given a desired folder path (base_folder), this function checks if a folder with that name exists.
    If it does, it appends a numeric suffix (_1, _2, etc.) until it finds a unique folder name,
    then returns the unique folder path.
    """
    if not os.path.exists(base_folder):
        return base_folder
    counter = 1
    while True:
        new_folder = f"{base_folder}_{counter}"
        if not os.path.exists(new_folder):
            return new_folder
        counter += 1

def process_evidence(threshold_percentage=80):
    """
    Processes all subfolders in the "Evidence" folder. In each subfolder, a folder called "Frames" is
    expected. All image files in each "Frames" folder are analyzed via the Plate Recognizer API.
    If the most common license plate meets the threshold percentage,
    the parent folder is renamed to that license plate number (ensuring the new name is unique).
    """
    regions = ['us-ia']  # Adjust region as needed
    authorization_token = 'Token 66494212dcac4174622a2ab25d3acd7a997e494f'
    evidence_folder = 'Evidence'

    if not os.path.isdir(evidence_folder):
        print(f"Evidence folder does not exist: {evidence_folder}")
        return

    subfolders = os.listdir(evidence_folder)
    for folder in subfolders:
        case_folder = os.path.join(evidence_folder, folder)
        if not os.path.isdir(case_folder):
            continue
        
        # Locate the "Frames" folder inside the case folder.
        frames_folder = os.path.join(case_folder, 'Frames')
        if not os.path.isdir(frames_folder):
            print(f"No 'Frames' folder in {case_folder}; skipping.")
            continue

        recognized_plates = []
        total_frames = 0

        print(f"\nProcessing frames in: {frames_folder}")
        for filename in os.listdir(frames_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_frames += 1
                image_path = os.path.join(frames_folder, filename)
                try:
                    with open(image_path, 'rb') as fp:
                        response = requests.post(
                            'https://api.platerecognizer.com/v1/plate-reader/',
                            data={'regions': regions},
                            files={'upload': fp},
                            headers={'Authorization': authorization_token}
                        )
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

                try:
                    json_response = response.json()
                except Exception as e:
                    print(f"Error parsing JSON for {filename}: {e}")
                    continue

                results = json_response.get('results', [])
                if results:
                    plate_number = results[0].get('plate', '').upper()
                    if plate_number:
                        recognized_plates.append(plate_number)
                        print(f"Plate for {filename}: {plate_number}")
                    else:
                        print(f"No plate detected in {filename}")
                else:
                    print(f"No plate detected in {filename}")
                time.sleep(1)  # Delay between API calls

        if total_frames == 0:
            print(f"No frames found in {frames_folder}. Removing {case_folder}...")
            shutil.rmtree(case_folder, ignore_errors=True)
            continue

        if not recognized_plates:
            print(f"No recognized plates in {frames_folder}. Removing {case_folder}...")
            shutil.rmtree(case_folder, ignore_errors=True)
            continue

        # Count the occurrences of each recognized plate.
        counter = Counter(recognized_plates)
        most_common_plate, most_common_count = counter.most_common(1)[0]
        recognized_count = len(recognized_plates)
        threshold_count = (threshold_percentage / 100.0) * recognized_count

        print(f"Results for {case_folder}: {most_common_plate} appeared {most_common_count} times out of {recognized_count} (threshold: {threshold_percentage}%).")

        if most_common_count >= threshold_count:
            # Define the new desired folder: evidence/{plate}
            new_case_folder = os.path.join(evidence_folder, most_common_plate)
            # Generate a unique folder name if a folder with the desired name exists
            unique_new_case_folder = get_unique_folder_name(new_case_folder)
            os.rename(case_folder, unique_new_case_folder)
            print(f"Renamed {case_folder} to: {unique_new_case_folder}")
        else:
            print(f"Threshold not met for {case_folder}. No renaming performed.")
            # Optionally, delete the folder
            # shutil.rmtree(case_folder, ignore_errors=True)

# Example usage:
process_evidence(threshold_percentage=10)