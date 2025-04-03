from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, send_file
from datetime import datetime
from report_org import generate_report
from natsort import natsorted
from flask_sqlalchemy import SQLAlchemy
from glob import glob
from metrics import dice_loss, dice_coef, iou
from tensorflow.keras.utils import CustomObjectScope
import logging
import secrets
import os
import random
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import nibabel as nib
from skimage.transform import resize
from cfgan2 import Generator

def save_results(image, y_pred, save_image_path):
    """ Overlay the predicted mask as bright red on the slice image. """
    
    # Ensure the predicted mask has the correct shape (H, W, 1)
    y_pred = np.expand_dims(y_pred, axis=-1)  # (H, W, 1)
    
    # Convert the prediction to RGB format (H, W, 3)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  # (H, W, 3)
    
    # Create a bright red mask (red = [255, 0, 0], green and blue = 0)
    red_mask = np.zeros_like(y_pred)
    red_mask[:, :, 0] = 255  # Set the red channel to 255 (bright red)
    
    # Ensure the original slice is in uint8 format (0-255)
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Overlap the red predicted mask on the original image
    # If the prediction is >0, place the red mask; otherwise, use the original image
    image_with_mask = np.where(y_pred > 0, red_mask, image)  # Replace with red mask where prediction > 0
    
    # Save the resulting image
    cv2.imwrite(save_image_path, image_with_mask)

num_classes = 5  # Update based on your model
axialresnet_path = "resnet_axial.pth"
coronalresnet_path="resnet_coronal.pth"
classes = ["buckle", "displaced", "no fracture" ,"non_displaced", "segmented"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads/CT_Scans'
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.String(5), unique=True, nullable=False)
    doctor_name = db.Column(db.String(80), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

    def __repr__(self):
        return f'<Doctor {self.username}>'

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(80), nullable=False)
    middle_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    gender = db.Column(db.String(80), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    diagnosis = db.Column(db.String(200), nullable=False)
    ct_scan = db.Column(db.String(120), nullable=False)
    doctor_id = db.Column(db.String(5), db.ForeignKey('doctor.doctor_id'), nullable=False)


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        doctor = Doctor.query.filter_by(username=username, password=password).first()
        if doctor:
            session['username'] = username
            session['doctor_id'] = doctor.doctor_id
            session['doctor_name'] = doctor.doctor_name
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('login'))
    else:
        return render_template('login.html')
    
@app.route('/fused_detection/<int:patient_id>')
def fused_detection(patient_id):
    patient_folder = os.path.join("uploads", "CT_Scans", f"CT{patient_id}")
    xray_ct_folder=os.path.join(patient_folder,"X-Ray_CT")
    os.makedirs(xray_ct_folder, exist_ok=True)
    for file in os.listdir(patient_folder):
        if file.endswith(".nii"):
            nii_file = os.path.join(patient_folder, file)
    ct_scan = nib.load(nii_file)
    ct_data = ct_scan.get_fdata()
    original_shape = ct_data.shape
    target_size = (256, 256)
    def normalize_image(img):
        """Normalize intensity to 0-255"""
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        return img.astype(np.uint8)

    def preprocess_and_resize(slice_data):
        """Resize & normalize a CT slice"""
        slice_data = np.nan_to_num(slice_data)  # Replace NaNs with 0
        slice_resized = resize(slice_data, target_size, anti_aliasing=True)
        return normalize_image(slice_resized)

    # ===== Generate Synthetic X-rays (MIP) =====
    mip_frontal = np.max(ct_data, axis=0)  # Axial Projection
    mip_side = np.max(ct_data, axis=1)  # Coronal Projection
    mip_alt_side = np.max(ct_data, axis=2)  # Sagittal Projection

    # Resize MIP images to match CT slices
    xray_frontal = preprocess_and_resize(mip_frontal)
    xray_side = preprocess_and_resize(mip_side)
    xray_alt_side = preprocess_and_resize(mip_alt_side)

    middle_x = ct_data.shape[0] // 2  # Axial (Top-down)
    middle_y = ct_data.shape[1] // 2  # Coronal (Front)
    middle_z = ct_data.shape[2] // 2  # Sagittal (Side)

    axial_slice = preprocess_and_resize(ct_data[middle_x, :, :])  # Axial slice
    coronal_slice = preprocess_and_resize(ct_data[:, middle_y, :])  # Coronal slice
    sagittal_slice = preprocess_and_resize(ct_data[:, :, middle_z])  # Sagittal slice

    cv2.imwrite(os.path.join(xray_ct_folder, "synthetic_xray_frontal.png"), xray_side)
    cv2.imwrite(os.path.join(xray_ct_folder, "synthetic_xray_side.png"), xray_frontal)
    cv2.imwrite(os.path.join(xray_ct_folder, "synthetic_xray_alt_side.png"), xray_alt_side)

    cv2.imwrite(os.path.join(xray_ct_folder, "axial_slice.png"), sagittal_slice)
    cv2.imwrite(os.path.join(xray_ct_folder, "coronal_slice.png"), coronal_slice)
    cv2.imwrite(os.path.join(xray_ct_folder, "sagittal_slice.png"), axial_slice)

    def load_generator(checkpoint_path, device):
        gen = Generator().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen.eval()  # Set to evaluation mode
        print(f"✅ Loaded model from {checkpoint_path}")
        return gen
    
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        return transform(img).unsqueeze(0)
    def enhance_image(image):
        # Convert to OpenCV format
        img_cv = np.array(image, dtype=np.uint8)
        
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_cv)
        
        # Apply Laplacian sharpening
        laplacian = cv2.Laplacian(img_clahe, cv2.CV_64F)
        img_sharp = cv2.convertScaleAbs(img_clahe - 0.5 * laplacian)
        
        # Apply Unsharp Masking
        blurred = cv2.GaussianBlur(img_sharp, (3, 3), 0)
        img_unsharp = cv2.addWeighted(img_sharp, 1.5, blurred, -0.5, 0)
        
        return Image.fromarray(img_unsharp)
    
    def fuse_images(xray_path, ct_path, gen, device, save_dir, name):
        os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

        # Load images
        xray = preprocess_image(xray_path).to(device)
        ct = preprocess_image(ct_path).to(device)

        # Save X-ray & CT images as reference
        xray_img = Image.open(xray_path).convert("L").resize((256, 256))
        ct_img = Image.open(ct_path).convert("L").resize((256, 256))
        
        # Generate fused image
        with torch.no_grad():
            fused = gen(xray, ct)
        
        assert isinstance(fused, torch.Tensor), "Generator output is not a tensor!"
        fused_img = fused.squeeze(0).cpu().numpy()

        # Normalize to [0,255] and convert to uint8
        fused_img = ((fused_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        
        # Apply blending to balance X-ray & CT
        alpha = 0.6  # Higher alpha gives more weight to the fused image
        fused_img = alpha * fused_img + (1 - alpha) * np.array(ct_img)
        fused_img = fused_img.clip(0, 255).astype(np.uint8)  # Ensure valid range
        if fused_img.ndim == 3:
            fused_img = fused_img.squeeze(0)
        # Apply enhancement filters
        fused_pil = Image.fromarray(fused_img, mode='L')
        fused_pil = enhance_image(fused_pil)
        
        # Save fused image
        fused_save_path = os.path.join(save_dir, name)
        fused_pil.save(fused_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    axial_checkpoint_path = "C:/Users/georg/FractureVision/cfgan_axial.pth"
    coronal_checkpoint_path = "C:/Users/georg/FractureVision/cfgan_coronal.pth"
    sagittal_checkpoint_path = "C:/Users/georg/FractureVision/cfgan_sagittal.pth"

    axial_gen = load_generator(axial_checkpoint_path, device)
    coronal_gen = load_generator(coronal_checkpoint_path, device)
    sagittal_gen = load_generator(sagittal_checkpoint_path, device)

    axial_xray_test = os.path.join(patient_folder,"X-ray_CT","synthetic_xray_alt_side.png")
    axial_ct_test = os.path.join(patient_folder,"X-ray_CT","axial_slice.png")
    coronal_xray_test = os.path.join(patient_folder,"X-ray_CT","synthetic_xray_frontal.png")
    coronal_ct_test = os.path.join(patient_folder,"X-ray_CT","coronal_slice.png")
    sagittal_xray_test = os.path.join(patient_folder,"X-ray_CT","synthetic_xray_side.png")
    sagittal_ct_test = os.path.join(patient_folder,"X-ray_CT","sagittal_slice.png")
    fused_images_folder = os.path.join(patient_folder,"fused_images")
    os.makedirs(fused_images_folder, exist_ok=True)

    fuse_images(axial_xray_test, axial_ct_test, axial_gen, device, fused_images_folder,"fused_axial.png")
    fuse_images(coronal_xray_test, coronal_ct_test, coronal_gen, device, fused_images_folder,"fused_coronal.png")
    fuse_images(sagittal_xray_test, sagittal_ct_test, sagittal_gen, device, fused_images_folder,"fused_sagittal.png")
    
    fused_results_folder=os.path.join(patient_folder,"fused_results")
    os.makedirs(fused_results_folder, exist_ok=True)
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        axial_model = tf.keras.models.load_model("axial.keras")
        coronal_model = tf.keras.models.load_model("coronal.keras")
        sagittal_model = tf.keras.models.load_model("sagittal.keras")

    axial_slice = os.path.join(fused_images_folder,"fused_axial.png")
    coronal_slice = os.path.join(fused_images_folder,"fused_coronal.png")
    sagittal_slice = os.path.join(fused_images_folder,"fused_sagittal.png")
    fused_slices=[axial_slice, coronal_slice, sagittal_slice]
    for path in fused_slices:
        name = os.path.basename(path).split(".")[0]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        slice = image / 255.0
        slice = np.expand_dims(slice, axis=0)
        if path==axial_slice:
            current_model = axial_model
        elif path==coronal_slice:
            current_model = coronal_model
        else:
            current_model = sagittal_model
        pred = current_model.predict(slice)[0]
        pred = np.squeeze(pred, axis=-1)
        pred = pred > 0.5
        pred = pred.astype(np.int32)
        predicted_class=""
        if(np.sum(pred) != 0):
            axialresnetmodel = models.resnet18()
            axialresnetmodel.fc = nn.Linear(axialresnetmodel.fc.in_features, num_classes)
            axialresnetmodel.load_state_dict(torch.load(axialresnet_path, map_location=device))
            axialresnetmodel = axialresnetmodel.to(device)
            axialresnetmodel.eval()
            def predict_image(image_path):
                image = Image.open(image_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

                with torch.no_grad():
                    outputs = axialresnetmodel(image)
                    _, preds = torch.max(outputs, 1)

                predicted_class = classes[preds.item()]
                return predicted_class
            predicted_class = predict_image(path)

        save_image_path = f"uploads/CT_Scans/CT{patient_id}/fused_results/{name}.png"
        save_results(image, pred, save_image_path)
        if predicted_class != "":
            # Load the image in color (BGR format)
            image = cv2.imread(save_image_path, cv2.IMREAD_COLOR)  # Load image in color (3 channels)

            # Create a figure and axis using Matplotlib
            fig, ax = plt.subplots()

            # Display the image in Matplotlib
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying in Matplotlib

            # Add text at the top-left corner
            ax.text(10, 20, predicted_class, color="white", fontsize=14, 
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

            # Hide axes for clean look
            ax.axis("off")

            # Save the image with the overlayed text (with tight bounding box and no padding)
            plt.savefig(save_image_path, bbox_inches="tight", pad_inches=0, dpi=300)

            # Close the plot to avoid memory issues
            plt.close(fig)

    return redirect(url_for('display', patient_id=patient_id, folder_type="fused_results"))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['doctor_name']
            username = request.form['username']
            password = request.form['password']
            
            # Check if the username already exists
            existing_doctor = Doctor.query.filter_by(username=username).first()
            if existing_doctor:
                flash('Username already exists. Please choose a different username.')
                return redirect(url_for('register'))
            
            # Generate a unique doctor ID
            doctor_id = "D" + str(random.randint(1000, 9999))
            while Doctor.query.filter_by(doctor_id=doctor_id).first() is not None:
                doctor_id = "D" + str(random.randint(1000, 9999))
            
            doctor = Doctor(username=username, password=password, doctor_id=doctor_id, doctor_name = name)
            db.session.add(doctor)
            db.session.commit()
            flash('Doctor successfully registered! Please log in.')
            return redirect(url_for('login'))
        except Exception as e:
            logging.error(f"Error during registration: {e}")
            flash('An error occurred during registration. Please try again.')
            return redirect(url_for('register'))
    else:
        return render_template('register.html')


@app.route('/home')
def home():
    username = session.get('username')
    name = session.get('doctor_name')
    doctor_id = session.get('doctor_id')
    if not username:
        return redirect(url_for('login'))
    patients = Patient.query.filter_by(doctor_id=doctor_id).all()
    return render_template('home.html', username=username, doctor_id=doctor_id, patients=patients, doctor_name = name)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('doctor_id', None)
    session.pop('doctor_name', None)
    return redirect(url_for('login'))

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        first_name = request.form['first_name']
        middle_name = request.form['middle_name']
        last_name = request.form['last_name']
        gender = request.form['gender']
        dob = datetime.strptime(request.form['dob'], '%Y-%m-%d')
        diagnosis = request.form['diagnosis']
        ct_scan = request.files['ct_scan']
        doctor_id = session.get('doctor_id')
        # Calculate age from dob
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        if ct_scan:
            filename = ct_scan.filename
            patient = Patient(first_name=first_name, middle_name=middle_name, last_name=last_name, gender=gender, dob=dob, age=age, diagnosis=diagnosis, ct_scan=filename, doctor_id=doctor_id)
            db.session.add(patient)
            db.session.commit()

            patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"CT{patient.id}")
            os.makedirs(patient_folder, exist_ok=True)

            patient_slices_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"CT{patient.id}", "Slices")
            os.makedirs(patient_slices_folder, exist_ok=True)
            
            axial_folder=os.path.join(patient_slices_folder,"Axial")
            coronal_folder=os.path.join(patient_slices_folder,"Coronal")
            sagittal_folder=os.path.join(patient_slices_folder,"Sagittal")
            os.makedirs(axial_folder, exist_ok=True)
            os.makedirs(coronal_folder, exist_ok=True)
            os.makedirs(sagittal_folder, exist_ok=True)
            
            file_path = os.path.join(patient_folder, filename)
            ct_scan.save(file_path)

            flash('Patient added successfully!')

            ct_scan = sitk.ReadImage(file_path)
            ct_scan_array = sitk.GetArrayFromImage(ct_scan)
            for i in range(ct_scan_array.shape[0]):
                axial_ct_slice = np.flipud(ct_scan_array[i, :, :])
                axial_output_filename = os.path.join(axial_folder, f"ct_slice_{i}.png")
                plt.imsave(axial_output_filename, axial_ct_slice, cmap='gray')
                a_slice=cv2.imread(axial_output_filename)
                a_slice=cv2.resize(a_slice, (256,256))
                plt.imsave(axial_output_filename,a_slice,cmap='gray')
            for i in range(ct_scan_array.shape[1]):
                coronal_ct_slice = np.flipud(ct_scan_array[:, i, :])
                coronal_output_filename = os.path.join(coronal_folder, f"ct_slice_{i}.png")
                plt.imsave(coronal_output_filename, coronal_ct_slice, cmap='gray')
                c_slice=cv2.imread(coronal_output_filename)
                c_slice=cv2.resize(c_slice, (256,256))
                plt.imsave(coronal_output_filename,c_slice,cmap='gray')
            for i in range(ct_scan_array.shape[2]):
                sagittal_ct_slice = np.flipud(ct_scan_array[:, :, i])
                sagittal_output_filename = os.path.join(sagittal_folder, f"ct_slice_{i}.png")
                plt.imsave(sagittal_output_filename, sagittal_ct_slice, cmap='gray')
                s_slice=cv2.imread(sagittal_output_filename)
                s_slice=cv2.resize(s_slice, (256,256))
                plt.imsave(sagittal_output_filename,s_slice,cmap='gray')

            return redirect(url_for('home'))
    return render_template('add_patient.html')

@app.route('/view_patients')
def view_patients():
    doctor_id = session.get('doctor_id')
    patients = Patient.query.filter_by(doctor_id=doctor_id).all()
    return render_template('view_patients.html', patients=patients)

@app.route('/display/<int:patient_id>/<string:folder_type>/')
def display(patient_id, folder_type):
    patient_folder = os.path.join("uploads", "CT_Scans", f"CT{patient_id}", folder_type)
    if folder_type=="fused_results":
        axial_image_paths=[f"/display/uploads/CT_Scans/CT{patient_id}/{folder_type}/fused_axial.png"]
        coronal_image_paths=[f"/display/uploads/CT_Scans/CT{patient_id}/{folder_type}/fused_coronal.png"]
        sagittal_image_paths=[f"/display/uploads/CT_Scans/CT{patient_id}/{folder_type}/fused_sagittal.png"]
        return render_template('display.html', axial_image_paths=axial_image_paths,
                           coronal_image_paths=coronal_image_paths,
                           sagittal_image_paths=sagittal_image_paths)
    else:
        if not os.path.exists(patient_folder) or not os.path.isdir(patient_folder):
            flash("No CT scan images found.")
            return redirect(url_for('view_patients'))
        axial_folder=os.path.join(patient_folder,"Axial")
        coronal_folder=os.path.join(patient_folder,"Coronal")
        sagittal_folder=os.path.join(patient_folder,"Sagittal")
        axial_image_paths = natsorted(
            [f"/display/uploads/CT_Scans/CT{patient_id}/{folder_type}/Axial/{file}"  
            for file in os.listdir(axial_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        coronal_image_paths = natsorted(
            [f"/display/uploads/CT_Scans/CT{patient_id}/{folder_type}/Coronal/{file}"  
            for file in os.listdir(coronal_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        sagittal_image_paths = natsorted(
            [f"/display/uploads/CT_Scans/CT{patient_id}/{folder_type}/Sagittal/{file}"  
            for file in os.listdir(sagittal_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

        return render_template('display.html', axial_image_paths=axial_image_paths,
                            coronal_image_paths=coronal_image_paths,
                            sagittal_image_paths=sagittal_image_paths)

@app.route('/detection/<int:patient_id>')
def detection(patient_id):

    patient_slices_folder = os.path.join("uploads", "CT_Scans", f"CT{patient_id}", "Slices")
    
    if os.path.exists(patient_slices_folder) and os.path.isdir(patient_slices_folder):
        patient_results_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"CT{patient_id}", "Results")
        os.makedirs(patient_results_folder, exist_ok=True)
        axial_results_folder=os.path.join(patient_results_folder,"Axial")
        coronal_results_folder=os.path.join(patient_results_folder,"Coronal")
        sagittal_results_folder=os.path.join(patient_results_folder,"Sagittal")
        os.makedirs(axial_results_folder, exist_ok=True)
        os.makedirs(coronal_results_folder, exist_ok=True)
        os.makedirs(sagittal_results_folder, exist_ok=True)
        
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            axial_model = tf.keras.models.load_model("axial.keras")
            coronal_model = tf.keras.models.load_model("coronal.keras")
            sagittal_model = tf.keras.models.load_model("sagittal.keras")
        
        axial_slices = sorted(glob(os.path.join("uploads", "CT_Scans", f"CT{patient_id}", "Slices", "Axial", "*")))
        coronal_slices = sorted(glob(os.path.join("uploads", "CT_Scans", f"CT{patient_id}", "Slices", "Coronal", "*")))
        sagittal_slices = sorted(glob(os.path.join("uploads", "CT_Scans", f"CT{patient_id}", "Slices", "Sagittal", "*")))
        
        for slice_path in axial_slices:
            name = os.path.basename(slice_path).split(".")[0]
            image = cv2.imread(slice_path, cv2.IMREAD_COLOR)
            slice = image / 255.0
            slice = np.expand_dims(slice, axis=0)

            pred = axial_model.predict(slice)[0]
            pred = np.squeeze(pred, axis=-1)
            pred = pred > 0.5
            pred = pred.astype(np.int32)
            predicted_class=""
            if(np.sum(pred) != 0):
                axialresnetmodel = models.resnet18()
                axialresnetmodel.fc = nn.Linear(axialresnetmodel.fc.in_features, num_classes)
                axialresnetmodel.load_state_dict(torch.load(axialresnet_path, map_location=device))
                axialresnetmodel = axialresnetmodel.to(device)
                axialresnetmodel.eval()
                def predict_image(image_path):
                    image = Image.open(image_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

                    with torch.no_grad():
                        outputs = axialresnetmodel(image)
                        _, preds = torch.max(outputs, 1)

                    predicted_class = classes[preds.item()]
                    return predicted_class
                predicted_class = predict_image(slice_path)

            save_image_path = f"uploads/CT_Scans/CT{patient_id}/Results/Axial/{name}.png"
            save_results(image, pred, save_image_path)
            
            # Assuming the `save_image_path` has been properly saved by `save_results` function
            if predicted_class != "":
                # Load the image in color (BGR format)
                image = cv2.imread(save_image_path, cv2.IMREAD_COLOR)  # Load image in color (3 channels)

                # Create a figure and axis using Matplotlib
                fig, ax = plt.subplots()

                # Display the image in Matplotlib
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying in Matplotlib

                # Add text at the top-left corner
                ax.text(10, 20, predicted_class, color="white", fontsize=14, 
                        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

                # Hide axes for clean look
                ax.axis("off")

                # Save the image with the overlayed text (with tight bounding box and no padding)
                plt.savefig(save_image_path, bbox_inches="tight", pad_inches=0, dpi=300)

                # Close the plot to avoid memory issues
                plt.close(fig)

        for slice_path in coronal_slices:
            name = os.path.basename(slice_path).split(".")[0]
            image = cv2.imread(slice_path, cv2.IMREAD_COLOR)
            slice = image / 255.0
            slice = np.expand_dims(slice, axis=0)

            pred = coronal_model.predict(slice)[0]
            pred = np.squeeze(pred, axis=-1)
            pred = pred > 0.5
            pred = pred.astype(np.int32)
            predicted_class=""
            if(np.sum(pred) != 0):
                coronalresnetmodel = models.resnet18()
                coronalresnetmodel.fc = nn.Linear(coronalresnetmodel.fc.in_features, num_classes)
                checkpoint = torch.load(coronalresnet_path, map_location=device)
                coronalresnetmodel.load_state_dict(checkpoint["model_state_dict"])
                coronalresnetmodel = coronalresnetmodel.to(device)
                coronalresnetmodel.eval()
                def predict_image(image_path):
                    image = Image.open(image_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

                    with torch.no_grad():
                        outputs = coronalresnetmodel(image)
                        _, preds = torch.max(outputs, 1)

                    predicted_class = classes[preds.item()]
                    return predicted_class
                predicted_class = predict_image(slice_path)

            save_image_path = f"uploads/CT_Scans/CT{patient_id}/Results/Coronal/{name}.png"
            save_results(image, pred, save_image_path)
            if predicted_class!="":
                # Load the image in color (BGR format)
                image = cv2.imread(save_image_path, cv2.IMREAD_COLOR)  # Load image in color (3 channels)

                # Create a figure and axis using Matplotlib
                fig, ax = plt.subplots()

                # Display the image in Matplotlib
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying in Matplotlib

                # Add text at the top-left corner
                ax.text(10, 20, predicted_class, color="white", fontsize=14, 
                        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

                # Hide axes for clean look
                ax.axis("off")

                # Save the image with the overlayed text (with tight bounding box and no padding)
                plt.savefig(save_image_path, bbox_inches="tight", pad_inches=0, dpi=300)

                # Close the plot to avoid memory issues
                plt.close(fig)


        for slice_path in sagittal_slices:
            name = os.path.basename(slice_path).split(".")[0]
            image = cv2.imread(slice_path, cv2.IMREAD_COLOR)
            slice = image / 255.0
            slice = np.expand_dims(slice, axis=0)

            pred = sagittal_model.predict(slice)[0]
            pred = np.squeeze(pred, axis=-1)
            pred = pred > 0.5
            pred = pred.astype(np.int32)

            save_image_path = f"uploads/CT_Scans/CT{patient_id}/Results/Sagittal/{name}.png"
            save_results(image, pred, save_image_path)

    return redirect(url_for('display', patient_id=patient_id, folder_type="Results"))
@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    patient = Patient.query.get(patient_id)
    if patient:
        db.session.delete(patient)
        db.session.commit()
        for i in os.listdir("uploads/CT_Scans"):
            id=int(i[2:])
            if id==patient_id:
                path=os.path.join("uploads/CT_Scans",i)
                print(path)
                shutil.rmtree(path)
        flash('Patient deleted successfully!')
    else:
        flash('Patient not found.')
    return redirect(url_for('view_patients'))

@app.route('/users')
def users():
    all_users = Doctor.query.all()
    return render_template('users.html', users=all_users)

@app.route('/display/uploads/<path:filename>')
def display_image(filename):
    return send_from_directory("uploads", filename)



from flask import send_from_directory
@app.route('/generate_report/<int:patient_id>', methods=['GET'])
def generate_report_route(patient_id):
    import time  # Import time for debugging (optional)

    patient = Patient.query.get(patient_id)
    if not patient:
        flash('Patient not found.')
        return redirect(url_for('view_patients'))

    doctor = Doctor.query.filter_by(doctor_id=patient.doctor_id).first()
    if not doctor:
        flash('Doctor not found.')
        return redirect(url_for('view_patients'))

    patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"CT{patient_id}")  
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam.jpg')  

    # Set the output path to the user's download folder
    output_path = os.path.join("C:\\Users\\georg\\Documents", f"report_{patient_id}.pdf")

    print(f"Output path: {output_path}")  # Debugging

    # Ensure patient folder exists
    print("11")
    if not os.path.exists(patient_folder):
        flash(f'Patient folder not found: {patient_folder}')
        
    print("2")
    # Generate the report
    print("1")
    generate_report(patient_folder, heatmap_path, output_path, patient, doctor)
    print("2")
    # Debugging: Check if the file was created
    if os.path.exists(output_path):
        print("PDF file created successfully!")
    else:
        print("PDF file was NOT created!")
        flash('Error generating report.')
        return redirect(url_for('view_patients'))

    # Return the file to the user
    return send_file(output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)