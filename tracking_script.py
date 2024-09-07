import cv2

# Full path to the video file
video_path = "C:/Users/91630/OneDrive/Desktop/PersonTrackingProject/test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Initialize the Haar Cascade for person detection
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize a dictionary to hold trackers for each detected person
trackers = {}
person_id = 0

# Initialize the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    persons = person_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in persons:
        tracker = cv2.TrackerCSRT_create()
        bbox = (x, y, w, h)
        tracker.init(frame, bbox)
        trackers[person_id] = tracker
        person_id += 1

    for pid, tracker in list(trackers.items()):
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {pid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            del trackers[pid]

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
