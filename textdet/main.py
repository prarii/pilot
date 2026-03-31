import cv2
import easyocr
import matplotlib.pyplot as plt
import os

def run_ocr(image_path, output_image="output.png", output_text="detected_text.txt", threshold=0.25):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Unable to read image.")
        return

    # Initialize OCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Perform OCR
    results = reader.readtext(img)

    extracted_text = []

    # Process results
    for (bbox, text, score) in results:
        if score >= threshold:
            # Convert bbox points to integers
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # Draw rectangle
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

            # Put detected text
            cv2.putText(img, text, top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            extracted_text.append(text)

    # Save output image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(output_image, bbox_inches='tight')
    print(f"[✓] Annotated image saved as '{output_image}'")

    # Save extracted text
    with open(output_text, "w", encoding="utf-8") as f:
        for line in extracted_text:
            f.write(line + "\n")

    print(f"[✓] Extracted text saved as '{output_text}'")


# ----------- MAIN -----------
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    run_ocr(image_path)
