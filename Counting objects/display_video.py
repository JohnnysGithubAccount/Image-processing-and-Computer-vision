import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy.ndimage import label, find_objects
from motpy import MultiObjectTracker
from motpy.core import setup_logger, Detection
import time
import argparse


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

# tracker arguments
parser.add_argument('--logging_file_name', type=str, help='Logging file name')
parser.add_argument('--tracker_max_staleness', type=int, default=12, help='Tracker max_staleness')

parser.add_argument('--show_video', default=False, action='store_true', help='Show the video as processing')
parser.add_argument('--save_video', default=False, action='store_true', help='Save video')

parser.add_argument('--resize', type=int, default=512, help='Resize the original image')
parser.add_argument('--circle_threshold', type=float, default=15.0, help='Minimum zero value pixel percent to be circle')

parser.add_argument('--active_tracks_min_steps_alive', type=int, default=7, help='Active tracks min steps alive')
parser.add_argument('--active_track_max_staleness', type=int, default=6, help='Active tracks max_staleness')

parser.add_argument('--order_pos', type=int, default=1, help='Order of the position model')
parser.add_argument('--dim_pos', type=int, default=2, help='Dimension of the space you are tracking')
parser.add_argument('--order_size', type=int, default=0, help='Similar to order_pos but size instead')
parser.add_argument('--dim_size', type=int, default=2, help='Dimension of the size')
parser.add_argument('--q_var_pos', type=float, default=5000., help='Process noise covariance for the position model')
parser.add_argument('--r_var_pos', type=float, default=0.1, help='Measurement noise covariance for the position model')

parser.add_argument('--min_iou', type=float, default=0.25, help='Minimum IoU to identify the same object')
parser.add_argument('--multi_match_min_iou', type=float, default=0.93,
                    help='Min IoU to classify the same object in case there multiple objects')

# process
parser.add_argument('--gaussian_blur_kernel', type=int, default=9, help='Kernel size for Gaussian blur')
parser.add_argument('--gaussian_blur_sigma', type=int, default=100, help='Sigma for Gaussian blur')

parser.add_argument('--kernel_closing1', type=int, default=2, help='Kernel size for closing 1')
parser.add_argument('--closing1_iter', type=int, default=11, help='Iterations for closing 1')

parser.add_argument('--kernel_opening', type=int, default=7, help='Kernel size for opening')
parser.add_argument('--opening_iter', type=int, default=3, help='Iterations for opening')

parser.add_argument('--kernel_closing2', type=int, default=2, help='Kernel size for closing 2')
parser.add_argument('--closing2_iter', type=int, default=17, help='Iterations for closing 2')

parser.add_argument('--name', type=str, default="22139057 Le Thanh Tai", help='Display name on the frame')

# The ArgumentParser.parse_args() method runs the parser and places the extracted data in an argparse
args = parser.parse_args()

logger = setup_logger(__name__, 'DEBUG', is_main=True, file_name=args.logging_file_name)
logger.info(f"Image size: {(args.resize, ) * 2}")
logger.info(f"Circle threshold: {args.circle_threshold}")
logger.info(f"{args.min_iou}")
logger.info(f"Using matching_fn_kwargs: 'min_iou': {args.min_iou}, 'multi_match_min_iou': {args.multi_match_min_iou}")
logger.info(f"Gaussian blur kernel size: {(args.gaussian_blur_kernel, ) * 2}")
logger.info(f"Gaussian blur sigma: {args.gaussian_blur_sigma}")
logger.info(f"Closing 1 kernel size: {(args.kernel_closing1, ) * 2}")
logger.info(f"Closing 1 iterations: {args.closing1_iter}")
logger.info(f"Opening kernel: {(args.kernel_opening, ) * 2}")
logger.info(f"Opening iterations: {args.opening_iter}")
logger.info(f"Closing 2 kernel size: {(args.kernel_closing2, ) * 2}")
logger.info(f"Closing 2 iterations: {args.closing2_iter}\n" + "-" * 100)

cap = cv2.VideoCapture("video/video3.mp4")
# get the video fps
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps
# saving frames for saving video
frames = []

# create MultiObjectTracker object
tracker = MultiObjectTracker(
    dt=dt,
    tracker_kwargs={'max_staleness': args.tracker_max_staleness},
    active_tracks_kwargs={'min_steps_alive': args.active_tracks_min_steps_alive,
                          'max_staleness': args.active_track_max_staleness},
    model_spec={'order_pos': args.order_pos,
                'dim_pos': args.dim_pos,
                'order_size': args.order_size,
                'dim_size': args.dim_size,
                'q_var_pos': args.q_var_pos,
                'r_var_pos': args.r_var_pos},
    matching_fn_kwargs={'min_iou': args.min_iou,
                        'multi_match_min_iou': args.multi_match_min_iou})

count = {
    'rectangle': 0,
    'circle': 0
}
ids = []


def calculate_zero_percentage(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    total_pixels = roi.size
    zero_pixel_count = np.sum(roi == 0)
    zero_percentage = (zero_pixel_count / total_pixels) * 100
    return zero_percentage


def main():
    while True:
        # read the frame from the video
        ret, frame = cap.read()

        # can't read frame from video means it's over
        if not ret:
            print("Can't receive frame")
            break

        # resize the frame
        frame = cv2.resize(frame, (args.resize, ) * 2)

        # turn the frame into gray scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur
        blurred = cv2.GaussianBlur(gray, (args.gaussian_blur_kernel, ) * 2, args.gaussian_blur_sigma)

        # find the threshold using thresh otsu
        threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # thresholding for segmenting objects
        binary = blurred > threshold
        binary = binary.astype(np.uint8)

        # first time closing -> fill zero-value pixels
        kernel = np.ones((args.kernel_closing1, ) * 2)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=args.closing1_iter)

        # opening -> clear extra one-value pixels
        kernel = np.ones((args.kernel_opening, ) * 2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=args.opening_iter)

        # second time closing -> fill holes created by opening
        kernel = np.ones((args.kernel_closing2, ) * 2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=args.closing2_iter)

        # labeling and find bounding box for objects
        labeled_array, num_features = label(closing)
        objects = find_objects(labeled_array)

        # bring the mask to range 0-255
        grayscale_image = closing * 255

        # Tracking objects
        detections = []
        for obj in objects:
            # Get coordinates of the bounding box
            ymin, xmin = obj[0].start, obj[1].start
            ymax, xmax = obj[0].stop, obj[1].stop
            # add object to the detections
            detections.append(Detection(box=np.array([xmin, ymin, xmax, ymax])))

        # Update tracker with detections
        t0 = time.time()
        active_tracks = tracker.step(detections=detections)
        elapsed = (time.time() - t0) * 1000.
        # logging elapsed time for tracking
        logger.debug(f'tracking elapsed time: {elapsed:.10} ms')

        # Draw detections and tracks
        for track in active_tracks:
            box = track.box
            zero_percentage = calculate_zero_percentage(grayscale_image, box)
            # calculate the percentage of zero-pixel in the bounding box

            # check if object haven't already existed, in the range 120 to 350 on the horizontal axis
            if track.id[:8] not in ids and (120 < int(box[0]) < 350):
                ids.append(track.id[:8])
                # count the object
                if zero_percentage >= args.circle_threshold:
                    count['circle'] += 1
                else:
                    count['rectangle'] += 1

            # display track id first 8 characters, show zero percentage calculated
            cv2.rectangle(grayscale_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, ), 2)
            cv2.putText(grayscale_image, f'{track.id[:8]}|{zero_percentage:.1f}', (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # display counts
            cv2.putText(grayscale_image, f'Circle: {count["circle"]} | Rectangle: {count["rectangle"]}', (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # draw boundaries line
            cv2.line(grayscale_image, (120, 0), (120, 512), (255, ), 1)
            cv2.line(grayscale_image, (350, 0), (350, 512), (255,), 1)

        # display sigma value used for gaussian blur
        cv2.putText(blurred, f'Sigma: {args.gaussian_blur_sigma}', (320, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # processing image for display final result
        display_img_upper = np.concatenate((gray, blurred), axis=1)
        display_img_lower = np.concatenate((binary * 255, grayscale_image), axis=1)

        display_img = np.concatenate((display_img_upper, display_img_lower), axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX  # font
        org = (10, 480)  # org
        fontScale = 1  # fontScale
        color = (255, )  # White color
        thickness = 2  # Line thickness of 2 px

        # Displaying names on the result
        display_img = cv2.putText(display_img, args.name,
                                  org, font, fontScale, color, thickness,
                                  cv2.LINE_AA)

        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        frames.append(display_img)

        logger.info(f'Num rectangle : {count["rectangle"]}')
        logger.info(f'Num circle : {count["circle"]}\n' + "-" * 100)

        if args.show_video:
            cv2.imshow('frame', display_img)
            if cv2.waitKey(5) == ord('q'):
                break

    # release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    # save video
    if args.save_video:
        clip = ImageSequenceClip(frames, fps=fps)  # Set desired fps
        clip.write_videofile("output_video.mp4", codec='libx264')  # Use 'libx264' for MP4


if __name__ == "__main__":
    main()
