import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import detector
from PIL import Image


# Initialize DeepSORT
tracker = DeepSort(max_age=5)
video_path = "children_playing_football.mp4"
# Open video capture
cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
 print("Cannot open camera")
 exit()


# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         15, size)


def change_format(results):
    #{'scores': tensor([0.2189, 0.2267]),
    # 'labels': tensor([1, 0]),
    # 'boxes': tensor([[213.5412,  83.6771, 308.0984, 119.0496],
     #  [171.6403, 136.5138, 269.0728, 251.2748]])} 

    # output expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()

    output = []
    for index, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        left = xmin
        top = ymin
        w = xmax - xmin
        h = ymax - ymin

        output.append(
            ([left, top, w, h], scores[index], labels[index])
        )
    return output


def run():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        classes = ["football",]
        class_description = ["a picture of a football"]
        
        # Detection using custom object detector
        detections = detector.detector(Image.fromarray(frame), class_description)

        tracks = tracker.update_tracks(change_format(detections), frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        for track in tracks:
            # print(track.is_confirmed())
            # if not track.is_confirmed():
            #     continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            # Visualize tracked objects
        
            x1, y1, x2, y2 = ltrb
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # write video
        result.write(frame) 

        cv2.imshow("output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()