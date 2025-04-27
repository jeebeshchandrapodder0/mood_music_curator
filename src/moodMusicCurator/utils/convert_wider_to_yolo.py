import os

def convert_wider_to_yolo(input_txt, image_base_dir, output_label_dir, img_size=(640, 480)):
    os.makedirs(output_label_dir, exist_ok=True)

    with open(input_txt, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        image_path = lines[idx].strip()
        num_faces = int(lines[idx + 1].strip())
        idx += 2

        label_rel_path = image_path.replace('.jpg', '.txt')
        label_path = os.path.join(output_label_dir, label_rel_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        with open(label_path, 'w') as out_label_file:
            for _ in range(num_faces):
                box = list(map(int, lines[idx].strip().split()))
                x, y, w, h = box[:4]
                # Convert to YOLO format
                x_center = (x + w / 2) / img_size[0]
                y_center = (y + h / 2) / img_size[1]
                width = w / img_size[0]
                height = h / img_size[1]
                out_label_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                idx += 1

    print(f"✅ Converted annotations saved to: {output_label_dir}")

# Example usage
if __name__ == "__main__":
    convert_wider_to_yolo(
        input_txt="src/moodMusicCurator/dataset/wider_face_split/wider_face_train_bbx_gt.txt",
        image_base_dir="src/moodMusicCurator/dataset/WIDER_train/images",
        output_label_dir="src/moodMusicCurator/dataset/labels/train",
        img_size=(640, 480)  # Approx. size for normalization
    )
