import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(input_dir, output_dir, classes_file):
    
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Classes File: {classes_file}")
    
    # Read class names
    with open(classes_file, 'r') as f:
        class_names = f.read().splitlines()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all annotation files
    for filename in os.listdir(input_dir):
        if not filename.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(input_dir, filename))
        root = tree.getroot()

        # Open output file
        output_file_path = os.path.join(output_dir, filename.replace('.xml', '.txt'))
        with open(output_file_path, 'w') as out_file:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue
                
                class_id = class_names.index(class_name)
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Convert to YOLO format
                image_width = int(root.find('size/width').text)
                image_height = int(root.find('size/height').text)
                x_center = (xmin + xmax) / (2.0 * image_width)
                y_center = (ymin + ymax) / (2.0 * image_height)
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                # Write to output file
                out_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def main():
    parser = argparse.ArgumentParser(description='Convert PascalVOC annotations to YOLO format.')
    parser.add_argument('input_dir', type=str, help='Path to input directory containing PascalVOC annotations.')
    parser.add_argument('output_dir', type=str, help='Path to output directory for YOLO format annotations.')
    parser.add_argument('classes_file', type=str, help='Path to the classes.txt file.')
    args = parser.parse_args()

    convert_voc_to_yolo(args.input_dir, args.output_dir, args.classes_file)

if __name__ == "__main__":
    main()
