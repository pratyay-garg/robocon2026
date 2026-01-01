import cv2
import numpy as np
import onnxruntime as ort

# --- CONFIGURATION ---
MODEL_PATH = "robocon_model.onnx"
CONFIDENCE_THRESHOLD = 0.8  # Model must be 80% sure of the symbol
MIN_BOX_AREA = 5000         # Ignore tiny red specks (noise)

# Class Names
CLASS_NAMES = ['Logo', 'Oracle', 'Random'] 

# --- LOAD MODEL ---
print("Loading ONNX Model...")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- COLOR RANGES (HSV) ---
# Red wraps around the color wheel (0-10 and 170-180)
LOWER_RED1 = np.array([0, 160, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 160, 70])
UPPER_RED2 = np.array([180, 255, 255])

def preprocess_for_model(img_roi):
    # 1. Convert to Grayscale
    
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Thresholding (Standard, NO Inversion)
    # This turns the Dark Gray background to BLACK (0).
    # This turns the White Symbol to WHITE (255).
    # Result: White Symbol on Black Background.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
   

    # 3. Convert back to 3 channels (RGB) for the model
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # 4. Normalize & Reshape
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0
    
    return img

def main():

    # Use 0 for default laptop cam, 1 for external
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("System Ready. Point at a Red Box.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- STEP 1: DETECT RED BOX (THE GATEKEEPER) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Combine both red ranges
        mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        red_mask = mask1 + mask2
        
        # Clean up noise (Morphology)
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find Contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_valid_box = False
        largest_box = None
        
        if contours:
            # Find the biggest red object
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > MIN_BOX_AREA:
                # Get Bounding Box (x, y, width, height)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # --- CHECK "ACCURACY" (Pixel Density) ---
                # This ensures it's a solid red box, not just a red wire or noise
                roi_mask = red_mask[y:y+h, x:x+w]
                red_pixel_count = cv2.countNonZero(roi_mask)
                total_area = w * h
                density = red_pixel_count / total_area
                
                # Only proceed if the box is sufficiently "Red" (>50% filled)
                
                if density > 0.5: 
                    found_valid_box = True
                    largest_box = (x, y, w, h)

        # --- DECISION TREE ---
        if found_valid_box:
            # 1. We have a Red Box
            x, y, w, h = largest_box
            
            # 2. Crop the image (Focus ONLY on the box)
            roi = frame[y:y+h, x:x+w]
            
            # 3. Send cropped ROI to Model
            try:
                input_tensor = preprocess_for_model(roi)
                outputs = session.run(None, {input_name: input_tensor})
                
                # 4. Interpret Result
                probs = outputs[0][0] # Softmax probabilities if added, or raw logits

                # Using simple argmax on logits works fine for classification
                prediction_idx = np.argmax(probs)
                
                # Calculate simple confidence (Softmax-ish)
                exp_scores = np.exp(probs - np.max(probs))
                softmax = exp_scores / np.sum(exp_scores)
                confidence = softmax[prediction_idx]
                
                final_class = CLASS_NAMES[prediction_idx]
                
                # Draw Green Box & Label
                color = (0, 255, 0) # Green
                label_text = f"{final_class} ({confidence:.2f})"
                
            except Exception as e:
                print(f"Inference Error: {e}")
                final_class = "Error"
                color = (0, 0, 255)
                
            # Draw on Frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        else:
            # --- NO RED BOX FOUND ---
            # Default to "Random"
            cv2.putText(frame, "Status: No Red Box (Random)", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show the view
        cv2.imshow('Robocon Vision: Red Box Gatekeeper', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()