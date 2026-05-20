import os
import argparse
import cv2

# ===================================================================

# parse command line arguments for paths to the data and model

parser = argparse.ArgumentParser(
    description='Perform image classification on x-ray images!')

parser.add_argument(
    "--data",
    type=str,
    help="specify path to the images",
    default='Results')

parser.add_argument(
    "--model",
    type=str,
    help="specify path to model weights",
    default='classifier.model')

args = parser.parse_args()

# ===================================================================

# load model weights:

model = cv2.dnn.readNetFromONNX(args.model)

# lists to keep filenames, images and identifiers for healthy and sick labels:

names = []
images = []
healthys = []
pneumonias = []

# the first 50 images are healthy and the next 50 are not:

for i in range(1, 51):
    healthys.append(f'im{str(i).zfill(3)}')

for i in range(51, 101):
    pneumonias.append(f'im{str(i).zfill(3)}')

# read all the images from the directory

for file in os.listdir(args.data):
    names.append(file)
names.sort()

# remove any extra files Mac might have put in there:

if ".DS_Store" in names:
    names.remove(".DS_Store")

# keeping track of the number of correct predictions for accuracy:
correct = 0
wrong_files = []  # 틀린 파일 저장

# main loop:
for filename in names:

    # read image:
    img = cv2.imread(os.path.join(args.data, filename))

    if img is not None:

        # pass the image through the neural network:
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (256, 256), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward()

        # identify what the predicted label is:
        if output > 0.5:
            predicted = "pneumonia"
            is_correct = filename.startswith(tuple(pneumonias))
        else:
            predicted = "healthy"
            is_correct = filename.startswith(tuple(healthys))

        if is_correct:
            correct += 1
            print(f'{filename}: {"pneumonia" if output > 0.5 else "healthy"} ✅')
        else:
            correct_label = "pneumonia" if filename.startswith(tuple(pneumonias)) else "healthy"
            wrong_files.append((filename, predicted, correct_label))
            print(f'{filename}: {predicted} ❌  (정답: {correct_label})')

# ===================================================================

# print final accuracy:
print(f'\nAccuracy is {correct / len(names):.4f}  ({correct}/{len(names)})')

# 틀린 파일 요약 출력
print(f'\n{"="*50}')
print(f'❌ 틀린 파일 총 {len(wrong_files)}개:')
print(f'{"="*50}')

healthy_as_pneumonia = [(f, p, c) for f, p, c in wrong_files if c == "healthy"]
pneumonia_as_healthy = [(f, p, c) for f, p, c in wrong_files if c == "pneumonia"]

if healthy_as_pneumonia:
    print(f'\n🔴 Healthy인데 Pneumonia로 잘못 예측 ({len(healthy_as_pneumonia)}개):')
    for fname, pred, correct_label in healthy_as_pneumonia:
        print(f'  - {fname}')

if pneumonia_as_healthy:
    print(f'\n🔵 Pneumonia인데 Healthy로 잘못 예측 ({len(pneumonia_as_healthy)}개):')
    for fname, pred, correct_label in pneumonia_as_healthy:
        print(f'  - {fname}')