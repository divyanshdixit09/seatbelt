from flask import Flask, render_template, Response, jsonify, request
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms.functional as F
from torchvision import transforms, models
from collections import OrderedDict
from ultralytics import YOLO
import re
import sys
import threading
import time
import base64
import json
sys.path.insert(0, '/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/parseq')
from strhub.data.module import SceneTextDataModule
sys.path.insert(0, '/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/CRAFT-pytorch')
import craft_utils
import imgproc
from craft import CRAFT
import os

app = Flask(__name__)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def bounding_box(points):
    points = points.astype(np.int16)
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def is_valid_plate(text_boxes):
    if not text_boxes:
        return False, ""
    ignored_words = {
        "IND", "INDIA", "GOVT", "GOVERNMENT", "CNG", "OF", "VEHICLE", "", "-", ".",
        "BHARAT", "TRANSPORT", "IN"
    }
    sorted_items = sorted(text_boxes, key=lambda item: min([point[0] for point in item["box"]]))
    cleaned_tokens = []
    for item in sorted_items:
        text = item["text"].strip().upper()
        if text in ignored_words:
            continue
        cleaned = re.sub(r"[^A-Z0-9]", "", text)
        if cleaned:
            cleaned_tokens.append(cleaned)
    sequence = ''.join(cleaned_tokens)
    strict_pattern = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}")
    bharat_pattern = re.compile(r"[0-9]{2}BH[0-9]{4}[A-Z]{0,2}")
    matches = strict_pattern.findall(sequence)
    if matches:
        best_match = sorted(matches, key=len, reverse=True)[0]
        return True, best_match
    bh_match = bharat_pattern.search(sequence)
    if bh_match:
        return True, bh_match.group()
    return False, ""

class PlateDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = YOLO("/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/numberplate.pt")
    def detect_plates(self, frame):
        results = self.model(frame)
        plates = []
        for result in results:
            if result.boxes.shape[0] > 0:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    plate_img = frame[y1:y2, x1:x2]
                    plates.append((plate_img, (x1, y1, x2, y2)))
        return plates

class TextDetector:
    def __init__(self, device='cuda'):
        self.args = {
            "text_threshold": 0.3,
            "link_threshold": 0.3,
            "low_text": 0.5,
            "canvas_size": 1280,
            "mag_ratio": 1.5,
            "cuda": device == 'cuda'
        }
        self.net = CRAFT()
        if self.args["cuda"]:
            self.net.load_state_dict(copyStateDict(torch.load("/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/craft_mlt_25k.pth")))
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        else:
            self.net.load_state_dict(copyStateDict(torch.load("/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/craft_mlt_25k.pth", map_location='cpu')))
        self.net.eval()
    def detect_text(self, image):
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, self.args["canvas_size"], 
            interpolation=cv2.INTER_LINEAR, 
            mag_ratio=self.args["mag_ratio"]
        )
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        if self.args["cuda"]:
            x = x.cuda()
        with torch.no_grad():
            y, feature = self.net(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, 
            self.args["text_threshold"], 
            self.args["link_threshold"], 
            self.args["low_text"], 
            False
        )
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        cropped_images = []
        valid_polys = []
        for i, bbs in enumerate(boxes):
            crop = bounding_box(bbs)
            cropped = image[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]
            if not any(dim_size == 0 for dim_size in cropped.shape):
                cropped_images.append(Image.fromarray(cropped))
                valid_polys.append(bbs)
        return cropped_images, valid_polys

class TextRecognizer:
    def __init__(self, device='cuda'):
        self.device = device
        self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
        self.parseq = self.parseq.to(device).eval()
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)
    def recognize_batch(self, cropped_images, max_batch_size=8):
        if not cropped_images:
            return [], []
        batch_size = len(cropped_images)
        num_batches = (batch_size + max_batch_size - 1) // max_batch_size
        all_texts = []
        all_scores = []
        for i in range(num_batches):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, batch_size)
            batch_images = cropped_images[start_idx:end_idx]
            batch_tensors = [F.to_tensor(F.resize(img, self.parseq.hparams.img_size)) for img in batch_images]
            batch_tensors = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                logits = self.parseq(batch_tensors)
                pred = logits.softmax(-1)
                labels, confidences = self.parseq.tokenizer.decode(pred)
                for j in range(end_idx - start_idx):
                    all_texts.append(labels[j])
                    all_scores.append(torch.mean(confidences[j]).item())
        return all_texts, all_scores

def load_seatbelt_models():
    model_cls = models.mobilenet_v2()
    model_cls.classifier[1] = torch.nn.Linear(model_cls.classifier[1].in_features, 2)
    model_cls.load_state_dict(torch.load("/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/working_code/mobilenetv2_finetuned.pt", map_location="cpu"))
    model_cls.eval()
    return model_cls

def load_vehicle_detection_model():
    model_vehicle = YOLO("yolov8x.pt")
    return model_vehicle

def load_person_detection_model():
    model_person = YOLO('yolov8x.pt')
    return model_person

def detect_cars(image, model_vehicle):
    results = model_vehicle(image)
    car_bboxes = []
    for result in results:
        if result.boxes.shape[0] > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 2:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    car_bboxes.append((x1, y1, x2, y2))
    return car_bboxes

def detect_seatbelt_inside_cars(image, model_person, model_cls, car_bboxes):
    label_map = {0: "noseatbelt", 1: "seatbelt"}
    color_map = {"seatbelt": (0, 255, 0), "noseatbelt": (0, 0, 255)}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    results = model_person(image)
    seatbelt_results = []
    for result in results:
        if result.boxes.shape[0] > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    conf = float(box.conf[0])
                    for cx1, cy1, cx2, cy2 in car_bboxes:
                        if x1 >= cx1 and y1 >= cy1 and x2 <= cx2 and y2 <= cy2:
                            person_crop = image[y1:y2, x1:x2]
                            try:
                                pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                                input_tensor = transform(pil_img).unsqueeze(0)
                                with torch.no_grad():
                                    output = model_cls(input_tensor)
                                    _, pred = torch.max(output, 1)
                                    label = label_map[pred.item()]
                            except:
                                label = "undetected"
                            color = color_map.get(label, (255, 255, 255))
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            seatbelt_results.append({"bbox": (x1, y1, x2, y2), "seatbelt_status": label, "confidence": conf})
                            break
    return image, seatbelt_results

def detect_anpr_only(image, plate_detector, text_detector, text_recognizer, confidence_threshold=0.5):
    plates = plate_detector.detect_plates(image)
    anpr_results = []
    for plate_img, bbox in plates:
        if plate_img.size == 0:
            continue
        cropped_images, polys = text_detector.detect_text(plate_img)
        if not cropped_images:
            continue
        texts, scores = text_recognizer.recognize_batch(cropped_images)
        text_boxes = []
        for txt, score, poly in zip(texts, scores, polys):
            if score > confidence_threshold:
                text_boxes.append({
                    "text": txt,
                    "score": score,
                    "box": [list(map(int, point)) for point in poly]
                })
        is_valid, plate_str = is_valid_plate(text_boxes)
        if is_valid:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, plate_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            anpr_results.append({
                "plate_number": plate_str,
                "bbox": bbox,
                "confidence": max([tb["score"] for tb in text_boxes])
            })
    return image, anpr_results

class StreamProcessor:
    def __init__(self):
        self.current_frame = None
        self.violations = []
        self.is_processing = False
        self.stats = {"cars_detected": 0, "plates_detected": 0, "violations": 0}
        
    def process_rtsp_stream(self, rtsp_url, device='cuda'):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        seatbelt_cls = load_seatbelt_models()
        vehicle_model = load_vehicle_detection_model()
        person_model = load_person_detection_model()
        plate_detector = PlateDetector(device)
        text_detector = TextDetector(device)
        text_recognizer = TextRecognizer(device)
        os.makedirs("violations", exist_ok=True)
        frame_count = 0
        self.is_processing = True
        
        while self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            car_bboxes = detect_cars(frame, vehicle_model)
            self.stats["cars_detected"] = len(car_bboxes)
            if car_bboxes:
                frame_sb, seatbelt_results = detect_seatbelt_inside_cars(frame, person_model, seatbelt_cls, car_bboxes)
            else:
                frame_sb = frame.copy()
                seatbelt_results = []
            frame_final, anpr_results = detect_anpr_only(frame_sb, plate_detector, text_detector, text_recognizer)
            self.stats["plates_detected"] = len(anpr_results)
            
            for sb in seatbelt_results:
                if sb["seatbelt_status"] == "noseatbelt":
                    x1, y1, x2, y2 = sb["bbox"]
                    for cx1, cy1, cx2, cy2 in car_bboxes:
                        if x1 >= cx1 and y1 >= cy1 and x2 <= cx2 and y2 <= cy2:
                            matched_plate = "unknown"
                            for anpr in anpr_results:
                                px1, py1, px2, py2 = anpr["bbox"]
                                if px1 >= cx1 and py1 >= cy1 and px2 <= cx2 and py2 <= cy2:
                                    matched_plate = anpr["plate_number"]
                                    break
                            car_crop = frame_final[cy1:cy2, cx1:cx2]
                            snapshot_path = f"violations/frame{frame_count}_{matched_plate}.jpg"
                            cv2.imwrite(snapshot_path, car_crop)
                            violation = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "plate": matched_plate,
                                "image_path": snapshot_path,
                                "frame_count": frame_count
                            }
                            self.violations.append(violation)
                            self.stats["violations"] = len(self.violations)
                            break
            self.current_frame = frame_final
            frame_count += 1
        cap.release()

processor = StreamProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    rtsp_url = request.json.get('rtsp_url', 'rtsp://admin:int12345@localhost:8567/Streaming/channels/101')
    if not processor.is_processing:
        thread = threading.Thread(target=processor.process_rtsp_stream, args=(rtsp_url,))
        thread.daemon = True
        thread.start()
        return jsonify({"status": "started", "message": "Stream processing started"})
    return jsonify({"status": "already_running", "message": "Stream already processing"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    processor.is_processing = False
    return jsonify({"status": "stopped", "message": "Stream processing stopped"})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if processor.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processor.current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify(processor.stats)

@app.route('/api/violations')
def get_violations():
    return jsonify(processor.violations[-10:])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
