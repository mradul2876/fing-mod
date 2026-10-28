# 🔍 Biometric Fingerprint Identification System

Deep Learning based Biometric Fingerprint Recognition system jo **SOCOFing** dataset par train kiya gaya hai. Isme PyTorch se 3-layer Convolutional Neural Network (SmallCNN) banaya gaya hai aur testing ke liye Streamlit web interface use hua hai.

---

## 📌 Project Overview
* **Domain:** Computer Vision & Biometric Security
* **Deep Learning Framework:** PyTorch & Torchvision
* **Web App Framework:** Streamlit
* **Dataset:** SOCOFing (Sokokoto Coventry Fingerprint Database)
* **Dataset Scope:** Person 1 se 50 (Total 4,554 images — Real + Altered variants)
* **Input Resolution:** 96 × 96 pixels (Grayscale)
* **Model Checkpoint:** `fingerprint_cnn_model.pth`

---

## 📁 Project Structure
* `fingerprint_cnn.ipynb` — Training & validation pipeline
* `app.py` — Streamlit web interface
* `fingerprint_cnn_model.pth` — Trained CNN model weights
* `Selected_50_Persons/` — Extracted 4,554 images (Person 1-50)
* `README.md` — Project documentation & presentation guide

---

## 🧠 Model Architecture (SmallCNN)
* **Input Layer:** (1, 96, 96) Grayscale tensor
* **Block 1:** Conv2D (1 to 32 filters) -> BatchNorm -> ReLU -> MaxPool (48x48)
* **Block 2:** Conv2D (32 to 64 filters) -> BatchNorm -> ReLU -> MaxPool (24x24)
* **Block 3:** Conv2D (64 to 128 filters) -> BatchNorm -> ReLU -> MaxPool (12x12)
* **Flatten:** 128 x 12 x 12 = 18,432 Features
* **Dense Layer:** Linear (18,432 to 256) -> ReLU -> Dropout (0.2)
* **Output Classifier:** Linear (256 to 50 identities)

---

## 💻 Code Walkthrough: Jupyter Notebook (`fingerprint_cnn.ipynb`)

### 1. Libraries & Hardware Setup
* **Point 1:** Dataset scanning ke liye `os` aur image processing ke liye `PIL.Image` use hua hai.
* **Point 2:** PyTorch (`torch`, `torch.nn`) neural layers construct karta hai.
* **Point 3:** `torchvision.transforms` images ko resize aur tensor normalize karta hai.
* **Point 4:** `torch.manual_seed(42)` se training results consistent aur reproducible rehte hain.
* **Point 5:** Agar system me NVIDIA GPU hai to CUDA select hota hai, warna default CPU select hota hai.

---

### 2. Data Filtering (Person 1 to 50) & Transforms
* **Point 6:** `NUM_CLASSES = 50` set karke training scope ko balance kiya gaya hai.
* **Point 7:** Filename parsing se Person ID extract hoti hai (`10__M...bmp` se `10`).
* **Point 8:** Sirf 1 se 50 IDs wali images select hoti hain (total 4,554 images).
* **Point 9:** Class labels `pid - 1` karke 0-indexed banaye gaye hain (`0` se `49`).
* **Point 10:** Transforms har photo ko 96×96 Grayscale tensor aur `[-1, 1]` normalize karte hain.

---

### 3. Data Split & Batches
* **Point 11:** 85% data model training ke liye aur 15% validation/testing ke liye divide hota hai.
* **Point 12:** DataLoaders 64-64 images ke batches banate hain aur training ke dauran shuffle karte hain.

---

### 4. Neural Network Architecture Definition
* **Point 13:** 3 Convolutional stages fingerprint ke fine biometric details extract karte hain.
* **Point 14:** `BatchNorm2d` layers training ko tezi se converge karti hain.
* **Point 15:** `MaxPool2d(2, 2)` spatial resolution ko reduce karta hai (96 -> 48 -> 24 -> 12).
* **Point 16:** `Flatten` 3D feature volume ko 18,432-dimensional flat vector banata hai.
* **Point 17:** `Linear(18432, 256)` primary biometric features compress karta hai.
* **Point 18:** `Dropout(0.2)` 20% neurons temporarily disable karke overfitting rokta hai.
* **Point 19:** Final linear layer 50 classes ke classification logits output karti hai.

---

### 5. Training Loop & Model Export
* **Point 20:** Multi-class classification ke liye `CrossEntropyLoss` use hota hai.
* **Point 21:** `Adam` optimizer 0.001 learning rate ke saath weights automatically optimize karta hai.
* **Point 22:** Har epoch me model train hota hai, loss backward propagate hoti hai aur accuracy calculate hoti hai.
* **Point 23:** `torch.save` model ke trained parameters ko `fingerprint_cnn_model.pth` me serialize kar deta hai.

---

## 🌐 Code Walkthrough: Web Interface (`app.py`)

* **Point 24:** `@st.cache_resource` trained weights ko RAM me cache karke rakhta hai taaki app fast chale.
* **Point 25:** `net.eval()` evaluation mode activate karta hai jisse dropout testing me apply na ho.
* **Point 26:** `file_uploader` widget user se fingerprint input leta hai.
* **Point 27:** `unsqueeze(0)` input tensor ko 4D batch shape `[1, 1, 96, 96]` me convert karta hai.
* **Point 28:** `softmax` function logits ko 0 se 100% probabilities me map karta hai.
* **Point 29:** `pred.item() + 1` predicted class index ko actual registered person ID (1 se 50) me convert karta hai.
* **Point 30:** `st.success` aur `st.metric` identified candidate aur confidence score screen par display karte hain.

---

## 🚀 How to Run & Demonstrate

* **Step 1:** Verify karein ki `fingerprint_cnn_model.pth` folder me maujood hai.
* Point 1: Yeh model weights ki trained checkpoint file hai jise load kiya jata hai.

* **Step 2:** Terminal me command run karein: `streamlit run app.py`
* Point 2: Is command se Streamlit ka local server start hota hai aur UI trigger hoti hai.

* **Step 3:** Browser me `http://localhost:8501` open karein.
* Point 3: Yeh port application ka frontend graphical dashboard launch karta hai.

* **Step 4:** `Selected_50_Persons` folder se koi bhi fingerprint `.BMP` image upload karein.
* Point 4: Web interface par file uploader component image load karta hai.

* **Step 5:** **Identify Person** button par click karein.
* Point 5: Background me model forward pass aur softmax confidence calculate karta hai.

* **Step 6:** Screen par matched person ID aur confidence percentage verify karein.
* Point 6: System identified identity ko green confirmation card me present karta hai.
*