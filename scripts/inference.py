import os
import cv2
import argparse
from ultralytics import YOLO

def perform_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        
        # Person Detection
        results = person_model(img_path)
        for bbox in results:
            x1, y1, x2, y2 = bbox['bbox']
            cropped_img = image[y1:y2, x1:x2]
            
            # PPE Detection on Cropped Image
            ppe_results = ppe_model(cropped_img)
            for ppe_bbox in ppe_results:
                px1, py1, px2, py2 = ppe_bbox['bbox']
                
                # Adjust ppe_bbox coordinates according to the cropped image
                cv2.rectangle(image, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 255, 0), 2)
                cv2.putText(image, f"{ppe_bbox['class']}: {ppe_bbox['confidence']:.2f}", 
                            (x1 + px1, y1 + py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, img_file), image)

def main():
    parser = argparse.ArgumentParser(description="Perform inference using YOLO models")
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('person_det_model', type=str, help='Path to person detection model weights')
    parser.add_argument('ppe_detection_model', type=str, help='Path to PPE detection model weights')
    args = parser.parse_args()

    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)

if __name__ == "__main__":
    main()
