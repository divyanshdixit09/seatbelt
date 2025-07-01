from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms, models
from collections import OrderedDict
import re
import sys
import time
import os
import json
import asyncio
from datetime import datetime
import glob
from pathlib import Path
import base64

sys.path.insert(0, '/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/parseq')
from strhub.data.module import SceneTextDataModule
sys.path.insert(0, '/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/CRAFT-pytorch')
import craft_utils
import imgproc
from craft import CRAFT

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
    ignored_words = {"IND", "INDIA", "GOVT", "GOVERNMENT", "CNG", "OF", "VEHICLE", "", "-", ".", "BHARAT", "TRANSPORT", "IN"}
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

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def draw_text_with_bg(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2, padding=5):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img

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
        self.args = {"text_threshold": 0.3, "link_threshold": 0.3, "low_text": 0.5, "canvas_size": 1280, "mag_ratio": 1.5, "cuda": device == 'cuda'}
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
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.args["canvas_size"], interpolation=cv2.INTER_LINEAR, mag_ratio=self.args["mag_ratio"])
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        if self.args["cuda"]:
            x = x.cuda()
        with torch.no_grad():
            y, feature = self.net(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.args["text_threshold"], self.args["link_threshold"], self.args["low_text"], False)
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

def load_all_models():
    speed_model = YOLO("/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/yolov8x.pt")
    seatbelt_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path="/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/yolov5s.pt")
    seatbelt_yolo.conf = 0.4
    seatbelt_yolo.classes = [0]
    seatbelt_cls = models.mobilenet_v2()
    seatbelt_cls.classifier[1] = torch.nn.Linear(seatbelt_cls.classifier[1].in_features, 2)
    seatbelt_cls.load_state_dict(torch.load("/media/mk/997211ec-8c91-4258-b58e-f144225899f4/3AI/classy-sort-yolov5/Honda_Speed/seatbelt_mobilenetv2.pth", map_location="cpu"))
    seatbelt_cls.eval()
    plate_detector = PlateDetector()
    text_detector = TextDetector()
    text_recognizer = TextRecognizer()
    return speed_model, seatbelt_yolo, seatbelt_cls, plate_detector, text_detector, text_recognizer

def process_seatbelt_anpr(crop_image, seatbelt_yolo, seatbelt_cls, plate_detector, text_detector, text_recognizer):
    label_map = {0: "noseatbelt", 1: "seatbelt"}
    color_map = {"seatbelt": (0, 255, 0), "noseatbelt": (0, 0, 255)}
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    results = seatbelt_yolo(crop_image)
    detections = results.xyxy[0]
    seatbelt_results = []
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        person_crop = crop_image[y1:y2, x1:x2]
        try:
            pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                output = seatbelt_cls(input_tensor)
                _, pred = torch.max(output, 1)
                label = label_map[pred.item()]
        except:
            label = "undetected"
        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(crop_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(crop_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        seatbelt_results.append({"bbox": (x1, y1, x2, y2), "seatbelt_status": label, "confidence": float(conf)})
    plates = plate_detector.detect_plates(crop_image)
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
            if score > 0.5:
                text_boxes.append({"text": txt, "score": score, "box": [list(map(int, point)) for point in poly]})
        is_valid, plate_str = is_valid_plate(text_boxes)
        if is_valid:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(crop_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(crop_image, plate_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            anpr_results.append({"plate_number": plate_str, "bbox": bbox, "confidence": max([tb["score"] for tb in text_boxes])})
    return crop_image, {"seatbelt_detections": seatbelt_results, "anpr_detections": anpr_results}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()
processing_active = False
models_loaded = False
speed_model, seatbelt_yolo, seatbelt_cls, plate_detector, text_detector, text_recognizer = None, None, None, None, None, None

@app.on_event("startup")
async def startup_event():
    global models_loaded, speed_model, seatbelt_yolo, seatbelt_cls, plate_detector, text_detector, text_recognizer
    if not os.path.exists('flagged_vehicles'):
        os.makedirs('flagged_vehicles')
    try:
        speed_model, seatbelt_yolo, seatbelt_cls, plate_detector, text_detector, text_recognizer = load_all_models()
        models_loaded = True
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/api/flagged-vehicles")
async def get_flagged_vehicles():
    try:
        flagged_dir = Path("flagged_vehicles")
        if not flagged_dir.exists():
            return {"vehicles": []}
        
        image_files = list(flagged_dir.glob("*.jpg")) + list(flagged_dir.glob("*.png"))
        vehicles = []
        
        for img_path in sorted(image_files, key=lambda x: x.stat().st_mtime, reverse=True):
            filename = img_path.name
            parts = filename.split('_')
            if len(parts) >= 4:
                vehicle_id = parts[1]
                frame_num = parts[2].replace('frame', '')
                timestamp = parts[3]
                speed = parts[6].replace('.jpg', '').replace('.png', '')
                
                vehicles.append({
                    "id": vehicle_id,
                    "filename": filename,
                    "timestamp": timestamp,
                    "speed": speed,
                    "frame": frame_num,
                    "path": f"/api/image/{filename}"
                })
        
        return {"vehicles": vehicles}
    except Exception as e:
        return {"vehicles": [], "error": str(e)}

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    image_path = Path("flagged_vehicles") / filename
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

@app.post("/api/start-processing")
async def start_processing():
    global processing_active
    if not models_loaded:
        return {"error": "Models not loaded"}
    processing_active = True
    asyncio.create_task(run_detection_system())
    return {"status": "started"}

@app.post("/api/stop-processing")
async def stop_processing():
    global processing_active
    processing_active = False
    return {"status": "stopped"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def run_detection_system():
    global processing_active
    rtsp_url = "rtsp://admin:int12345@localhost:8573/profile1"

    SOURCE = np.array([[1644, 460], [2560, 505], [2110, 1436], [0, 1067]])
    TARGET = np.array([[0, 0], [6, 0], [6, 15], [0, 15]])
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        await manager.broadcast(json.dumps({"type": "error", "message": "Could not open video stream"}))
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    byte_track = sv.ByteTrack()
    vehicle_class_ids = [2, 3, 5, 7]
    polygon = np.array([[1644, 460], [2560, 505], [2110, 1436], [0, 1067]], dtype=np.int32)
    zone = sv.PolygonZone(polygon=polygon)
    coordinates = defaultdict(lambda: {"history": deque(maxlen=fps * 3)})
    frame_count = 0

    x, y, w, h = cv2.boundingRect(polygon)
    loop = asyncio.get_event_loop()

    while processing_active:
        ret, frame = cap.read()
        if not ret:
            await manager.broadcast(json.dumps({"type": "error", "message": "Stream interrupted"}))
            break

        frame_count += 1

        # 🔁 Send preview frame on alternate frames
async def run_detection_system():
    global processing_active
    rtsp_url = "rtsp://admin:int12345@localhost:8573/profile1"

    SOURCE = np.array([[1370, 100], [1864, 100], [1724, 1062], [0, 902]])
    TARGET = np.array([[0, 0], [6, 0], [6, 25], [0, 25]])
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        await manager.broadcast(json.dumps({"type": "error", "message": "Could not open video stream"}))
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    byte_track = sv.ByteTrack()
    vehicle_class_ids = [2, 3, 5, 7]
    polygon = np.array([[1370, 100], [1864, 100], [1724, 1062], [0, 902]], dtype=np.int32)
    zone = sv.PolygonZone(polygon=polygon)
    coordinates = defaultdict(lambda: {"history": deque(maxlen=fps * 1)})
    frame_count = 0

    x, y, w, h = cv2.boundingRect(polygon)
    loop = asyncio.get_event_loop()
    latest_detections = []

    while processing_active:
        ret, frame = cap.read()
        if not ret:
            await manager.broadcast(json.dumps({"type": "error", "message": "Stream interrupted"}))
            break

        frame_count += 1

        # Preview frame every other frame
        if frame_count % 2 != 0:
            preview_frame = frame.copy()
            cv2.polylines(preview_frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

            if hasattr(latest_detections, 'tracker_id') and hasattr(latest_detections, 'xyxy'):
                for i in range(len(latest_detections)):
                    tracker_id = latest_detections.tracker_id[i]
                    if tracker_id is None:
                        continue
                    x1, y1, x2, y2 = map(int, latest_detections.xyxy[i])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    history = coordinates[tracker_id]["history"]
                    if len(history) >= 2:
                        dist = abs(history[-1] - history[0])
                        time_sec = len(history) / fps
                        speed_kph = int((dist / time_sec) * 3.6) if time_sec > 0 else 0
                        cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(preview_frame, f"{speed_kph} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            try:
                resized = cv2.resize(preview_frame, (640, 360))
                _, jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
                await manager.broadcast(json.dumps({"type": "frame", "data": frame_b64}))
            except:
                pass

            await asyncio.sleep(0.01)
            continue


        # Detection and tracking
        roi = frame[y:y + h, x:x + w]
        result = await loop.run_in_executor(None, speed_model, roi)
        detections = sv.Detections.from_ultralytics(result[0])
        mask_classes = [cls in vehicle_class_ids for cls in detections.class_id]
        detections = detections[mask_classes]
        if len(detections) == 0:
            latest_detections = []
            await asyncio.sleep(0.01)
            continue

        detections.xyxy[:, [0, 2]] += x
        detections.xyxy[:, [1, 3]] += y
        detections = byte_track.update_with_detections(detections)
        latest_detections = detections

        xyxy = detections.xyxy
        centroids = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2))
        world_coords = view_transformer.transform_points(centroids)
        in_zone_mask = zone.trigger(detections)

        for i, coord in enumerate(world_coords):
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else 'N/A'
            y_world = coord[1]
            coordinates[tracker_id]["history"].append(y_world)
            history = coordinates[tracker_id]["history"]

            if len(history) >= 2:
                dist = abs(history[-1] - history[0])
                time_sec = len(history) / fps
                speed_kph = int((dist / time_sec) * 3.6) if time_sec > 0 else 0
            else:
                speed_kph = 0

            if speed_kph > 20 and in_zone_mask[i]:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    processed_crop, _ = process_seatbelt_anpr(crop.copy(), seatbelt_yolo, seatbelt_cls, plate_detector, text_detector, text_recognizer)
                    timestamp = datetime.now().strftime("%Y%m%d_%f")[:-3]
                    filename = f"vehicle_{tracker_id}_frame{frame_count}_{timestamp}_{speed_kph}kmh.jpg"
                    filepath = f"flagged_vehicles/{filename}"
                    cv2.imwrite(filepath, processed_crop)

                    await manager.broadcast(json.dumps({
                        "type": "new_vehicle",
                        "data": {
                            "id": int(tracker_id) if tracker_id != 'N/A' else "N/A",
                            "filename": filename,
                            "timestamp": timestamp,
                            "speed": f"{speed_kph}kmh",
                            "frame": frame_count,
                            "path": f"/api/image/{filename}"
                        }
                    }))

        await asyncio.sleep(0.01)

    cap.release()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
