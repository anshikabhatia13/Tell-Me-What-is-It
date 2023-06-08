import cv2
import pyttsx3
import numpy as np

# Load the pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide the index if multiple webcams are available

# Initialize text-to-speech engine
engine = pyttsx3.init()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set the input blob for the network
    net.setInput(blob)
    
    # Run forward pass to get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    
    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output layer
    for out in outs:
        # Process each detection
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Minimum confidence threshold
                # Scale the bounding box coordinates relative to the size of the frame
                width = frame.shape[1]
                height = frame.shape[0]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Calculate top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Append the bounding box, confidence, and class ID to the respective lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Initialize list for detected objects
    detected_objects = []
    
    # Draw the bounding boxes and labels on the frame, and add object labels to the detected_objects list
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detected_objects.append(label)
    
    # Convert detected_objects list to a sentence
    detected_objects_sentence = ', '.join(detected_objects)

    # Provide audio feedback using text-to-speech
    engine.say(f"Ankish, here: {detected_objects_sentence}")
    
    engine.runAndWait()

    # Display the frame
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(5)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Release the capture and destroy windows
cap.release()

cv2.destroyAllWindows()


