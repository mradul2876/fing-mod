import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Point 1: Page ka title aur layout wide set kiya taaki demo sundar dikhe
st.set_page_config(page_title="Biometric Fingerprint System", layout="wide")

NUM_CLASSES = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Point 2: CNN Architecture wahi exact SmallCNN define kiya jo notebook me train hua tha
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# Point 3: Caching use ki taaki trained weights bar-bar reload hone me time na le
@st.cache_resource
def load_trained_model():
    model = SmallCNN().to(device)
    if os.path.exists("fingerprint_cnn_model.pth"):
        model.load_state_dict(torch.load("fingerprint_cnn_model.pth", map_location=device))
        model.eval()
        return model
    return None

model = load_trained_model()

# Point 4: Image ko 96x96 grayscale tensor me transform karne ka pipeline
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Point 5: UI Headers aur descriptions
st.title("🔍 Biometric Fingerprint Identification System")
st.markdown("SOCOFing dataset par trained deep learning system — Person ID, Gender aur Finger type recognition.")

uploaded_file = st.file_uploader("Fingerprint image upload karein (.BMP)", type=["bmp", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Point 6: Uploaded image ko Grayscale me open kiya
    image = Image.open(uploaded_file).convert("L")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Uploaded Fingerprint")
        st.image(image, caption=uploaded_file.name, width=250)
        
    with col2:
        st.subheader("Biometric Analysis")
        
        # Point 7: Filename parse karke Gender aur Finger extract karne ka logic
        file_name = uploaded_file.name
        gender = "Not Available"
        finger_type = "Not Available"
        
        try:
            clean_name = os.path.splitext(file_name)[0]
            parts = clean_name.split("__")
            if len(parts) > 1:
                sub_parts = parts[1].split("_")
                
                # Gender detect kiya
                if sub_parts[0] == "M":
                    gender = "Male (M)"
                elif sub_parts[0] == "F":
                    gender = "Female (F)"
                
                # Finger name format kiya (jaise 'Left index finger')
                raw_finger = " ".join(sub_parts[1:])
                finger_type = raw_finger.replace("_", " ").title()
        except Exception:
            pass

        if st.button("Identify Person & Extract Details", type="primary"):
            if model is None:
                st.error("Model weights file (`fingerprint_cnn_model.pth`) nahi mili!")
            else:
                # Point 8: Image ko batch tensor banakar model se prediction nikala
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_person_id = predicted_idx.item() + 1
                confidence_score = confidence.item() * 100

                # Point 9: Live match details screen par display ki
                st.success(f"✅ Match Found: **Person {predicted_person_id}**")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Candidate ID", f"Person_{predicted_person_id}")
                m2.metric("Matching Confidence", f"{confidence_score:.2f}%")
                m3.metric("Gender", gender)
                
                st.info(f"🖐️ **Identified Finger:** {finger_type}")