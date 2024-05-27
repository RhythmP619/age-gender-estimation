import torch
import cv2
import torch.nn as nn
from torchvision import transforms
from model import AgeEstimator, GenderEstimator
from facenet_pytorch.models.mtcnn import MTCNN

IMG_HEIGHT, IMG_WIDTH = 180, 180
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=DEVICE)
age_model = AgeEstimator().to(DEVICE)
age_model.load_state_dict(torch.load('age_model.pth'))
age_model.eval()

gender_model = GenderEstimator().to(DEVICE)
gender_model.load_state_dict(torch.load('gender_model.pth'))
gender_model.eval()

def preprocess_img(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
    ])
    return preprocess(image).unsqueeze(0)

img_path = 'test.jpg'  # Replace 'path_to_your_image.jpg' with the path to your image
img = cv2.imread(img_path)

# Detect faces in the image
boxes, probs = mtcnn.detect(img)
if boxes is not None:
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = img[y1:y2, x1:x2]
        preprocessed_face = preprocess_img(face).to(DEVICE)
        with torch.no_grad():
            age_output = age_model(preprocessed_face.to(DEVICE))
            age = age_output.item()
            gender_output = gender_model(preprocessed_face.to(DEVICE))
            gender = torch.argmax(gender_output).item()
            gender_text = "Male" if gender == 0 else "Female"
            
        cv2.putText(img, f"Age: {age:.1f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"Gender: {gender_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
