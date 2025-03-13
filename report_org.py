from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from datetime import datetime
import os
from glob import glob

def generate_report(patient_folder, heatmap_image_path, output_path, patient, doctor):
    patient_folder = os.path.abspath(patient_folder)  # Convert to absolute path
    results_folder = os.path.join(patient_folder, "Results")  # Look inside 'Results/' subfolder
    print(f"🔍 Searching Images Inside: {results_folder}")

    c = canvas.Canvas(output_path)

    def add_footer():
        """Adds the blue footer bar and date/time at the bottom of every page."""
        # Draw the Blue Footer Bar
        c.setFillColor(HexColor("#5e7dac"))
        c.roundRect(10, 5, 575, 15, 5, stroke=0, fill=1)

        # Draw the Date & Time on top of the footer
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.white)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(175, 8, f"Report generated on: {current_time}")

    def create_first_page():
        c.setFillColor(HexColor("#ffffff"))
        c.rect(0, 0, 600, 900, stroke=0, fill=1)

        # Header
        c.setFillColor(HexColor("#5e7dac"))
        c.roundRect(10, 5, 575, 15, 5, stroke=0, fill=1)
        c.roundRect(-30, 750, 5000000, 595, 200, stroke=0, fill=1)

        c.setFont("Helvetica-Bold", 36)
        c.setFillColor(colors.white)
        c.drawString(220, 785, "REPORT")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(430, 802, "FractureVision")
        c.drawString(440, 785, "Hospital")
        c.drawImage("H.jpg", 520, 769, height=50, width=50)

        # Patient details
        c.setFillColor(HexColor("#5e7dac"))
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, 700, "Patient Details:")
        c.setFillColor(colors.black)
        c.rect(50, 420, 500, 230, stroke=1, fill=0)

        c.setFont("Helvetica", 16)
        patient_details = [
            ("First Name:", patient.first_name),
            ("Last Name:", patient.last_name),
            ("Gender:", patient.gender),
            ("Date of Birth:", patient.dob.strftime('%Y-%m-%d')),
            ("Age:", patient.age),
            ("Diagnosis:", patient.diagnosis)
        ]

        y_position = 620
        for label, value in patient_details:
            c.drawString(60, y_position, label)
            c.drawString(200, y_position, str(value))
            y_position -= 30

        # Doctor details
        c.setFillColor(HexColor("#5e7dac"))
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, 300, "Doctor Details:")
        c.setFillColor(colors.black)
        c.rect(50, 160, 500, 85, stroke=1, fill=0)

        c.setFont("Helvetica", 16)
        doctor_details = [
            ("Doctor ID:", doctor.doctor_id),
            ("Doctor Name:", doctor.doctor_name)
        ]

        y_position = 210
        for label, value in doctor_details:
            c.drawString(60, y_position, label)
            c.drawString(200, y_position, str(value))
            y_position -= 30

        add_footer()  # Add date and time
        c.showPage()  # Move to next page for images

    def top_bar():
        c.setFillColor(HexColor("#5e7dac"))
        c.roundRect(10, 5, 575, 15, 5, stroke=0, fill=1)
        c.roundRect(-30, 750, 5000000, 595, 200, stroke=0, fill=1)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(150, 795, "CT Scan Images")

    

        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.white)
        c.drawString(430, 802, "FractureVision")
        c.drawString(440, 785, "Hospital")
        c.drawImage("H.jpg", 520, 769, height=50, width=50)
    

    def add_images(image_list):
        images_per_page = 3
        x_positions = [50, 50, 50]
        y_positions = [520, 280, 70]
       

        for i, image_path in enumerate(image_list):
            if i % images_per_page == 0:
                if i != 0:
                    top_bar()
                    add_footer()  # Add footer before starting a new page
                    c.showPage()  # Start a new page
               

            img_x = x_positions[i % images_per_page]
            img_y = y_positions[i % images_per_page]

            try:
                c.drawImage(image_path, img_x, img_y, width=500, height=200)
                print(f"Added Image: {image_path}")
            except Exception as e:
                print(f"Error adding image {image_path}: {e}")

        add_footer()  # Add footer on the last page

    # Generate first page
    create_first_page()

    # Fetch images inside the 'Results/' subfolder
    image_files = sorted(glob(os.path.join(results_folder, "*.jpg")) + 
                         glob(os.path.join(results_folder, "*.jpeg")) + 
                         glob(os.path.join(results_folder, "*.png")))

    # Debugging: Print all images found
    print(f"Images found in 'Results': {image_files}")

    # Ensure file paths are valid
    image_files = [img.replace("\\", "/") for img in image_files if os.path.exists(img)]

    # Add images to the report
    if image_files:
        add_images(image_files)
    else:
        c.setFont("Helvetica-Bold", 18)
        c.setFillColor(colors.red)
        c.drawString(50, 700, "No CT images available in 'Results'.")
        add_footer()  # Add footer even if there are no images
        print("No images found in 'Results' folder.")

    # Save the final PDF
    try:
        c.save()
        print(f"Report saved successfully: {output_path}")
    except PermissionError as e:
        print(f"Permission Error: {e}")
        print("Try running the script as Administrator or changing file permissions.")