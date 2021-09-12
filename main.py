import cv2
import sys
import time


def get_args():
  if len(sys.argv) < 3:
    exit(-1)

  return int(sys.argv[1]), int(sys.argv[2])


def handle_contours(cutted_frame, obj_area):
  # apply the musk only in the cutted frame
  mask = object_detector.apply(cutted_frame)
  _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > obj_area:
      x, y, w, h = cv2.boundingRect(cnt)
      cv2.rectangle(cutted_frame, (x, y), (x + w, y + h),  (0, 255, 0), 3)

  return mask


def draw_all(frame, center_x, center_y):
  cv2.rectangle(frame, (center_x + 100, center_y + 100), (center_x + 600, center_y + 350), (0, 0, 255), 2)


def process_frame(frame, obj_area):
  h_f, w_f, _ = frame.shape
  
  center_x, center_y = int(w_f/2), int(h_f/2)
  cutted_frame = frame[center_y + 100:center_y+350, center_x + 100: center_x + 600]
  mask = handle_contours(cutted_frame, obj_area)
  draw_all(frame, center_x, center_y)
  

  cv2.imshow('Video', frame)
  cv2.imshow('Mask', mask)
  cv2.imshow('Object', cutted_frame)
  # cv2.resizeWindow('Video', w_f, h_f)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
      exit(0)


if __name__ == '__main__':
  obj_area, fps = get_args()
  video = cv2.VideoCapture("test-video.mp4")
  object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)

  prev = 0

  while True:
    time_elapsed = time.time() - prev
    can_show, frame = video.read()

    if time_elapsed > 1./fps and can_show:
      prev = time.time()
      process_frame(frame, obj_area)
  
  video.release()
  cv2.destroyAllWindows()
