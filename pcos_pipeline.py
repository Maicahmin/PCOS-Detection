# IMPORT LIBRARIES
# Liraries Needed for the machine

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# INITIALIZATION FUNCTION
# A class groups all related functions into one object
# Function runs when the pipeline start

class PCOSPipeline:

    def __init__(self, config):
        self.config = config
        self.model = None

# SETUP
# This Loads the YOLO12 model
    
    def setup(self):
        self.model = YOLO(self.config["weights"])

# TRAINING
# This Trains the model using PCOS ultrasound Dataset
    
    def train(self):
        self.model.train(
            data=self.config["data_path"],
            epochs=self.config["epochs"],
            imgsz=self.config["imgsz"],
            batch=self.config["batch"],
            patience=self.config["patience"],
            device=self.config["device"],
            save=True
        )

# EVALUATION
# This checks how well the model performs on validation data

    def evaluate(self):
        return self.model.val()

# INFERENCE AND VISUALIZATIONN

    def infer(self): # This runs the trained model on new images
        if self.config["test_image"]:
            results = self.model(self.config["test_image"]) # Run Detection 
            for r in results:
                img = cv2.imread(self.config["test_image"]) # Read Images
                boxes = r.boxes.xyxy.cpu().numpy() # Extract Detection data
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names = self.model.names

                for box, score, cls_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls_id]} {score:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2) # Draw Bounding Boxes
                    cv2.putText(img, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255,255,255), 2)

                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Display the image
                plt.axis("off")
                plt.show()

# RUN FUNCTION
# This is the main controller of the pipeline

    def run(self):
        self.setup()
        self.train()
        self.evaluate()
        self.infer()
