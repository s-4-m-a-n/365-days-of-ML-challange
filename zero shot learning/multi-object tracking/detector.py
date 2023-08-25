from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

model_path = "../object detection/saved_owlvit_model/model"

#  Load saved model
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)


# detector
def detector(frame, class_description):
    with torch.no_grad():
        inputs = processor(text=class_description, images=frame, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([frame.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
    return results
