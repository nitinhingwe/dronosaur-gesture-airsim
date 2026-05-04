import cv2
import numpy as np
import time
from pathlib import Path

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    FormatType,
    HailoStreamInterface,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "hef_models" / "person_detection.hef"


def parse_detections(output, conf_threshold=0.45):
    detections = []

    def walk(obj, class_id=0):
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                walk(item, idx)
            return

        arr = np.asarray(obj)
        if arr.size == 0:
            return

        if arr.ndim == 1 and arr.shape[0] >= 5:
            rows = arr.reshape(1, -1)
        elif arr.ndim >= 2 and arr.shape[-1] >= 5:
            rows = arr.reshape(-1, arr.shape[-1])
        else:
            return

        for det in rows:
            x1, y1, x2, y2, score = det[:5]

            if float(score) < conf_threshold:
                continue

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "score": float(score),
                "class": int(class_id),
            })

    walk(output)
    return detections


def choose_best_person(detections):
    # COCO class 0 = person for normal YOLO models
    person_dets = [d for d in detections if d["class"] == 0]

    if not person_dets:
        return None

    best = None
    for det in person_dets:
        width = max(0.0, det["x2"] - det["x1"])
        height = max(0.0, det["y2"] - det["y1"])
        area = width * height

        # score + area helps avoid tiny noisy boxes
        rank = det["score"] * 0.7 + area * 0.3

        candidate = dict(det)
        candidate["area"] = area
        candidate["rank"] = rank

        if best is None or candidate["rank"] > best["rank"]:
            best = candidate

    return best


def smooth_box(prev, current, alpha=0.22):
    if current is None:
        return None

    if prev is None:
        return dict(current)

    smoothed = dict(current)

    for key in ["x1", "y1", "x2", "y2"]:
        smoothed[key] = alpha * current[key] + (1 - alpha) * prev[key]

    smoothed["score"] = current["score"]
    smoothed["class"] = current["class"]
    smoothed["area"] = (smoothed["x2"] - smoothed["x1"]) * (smoothed["y2"] - smoothed["y1"])

    return smoothed

def clamp_box(box):
    if box is None:
        return None

    box["x1"] = min(max(box["x1"], 0.0), 1.0)
    box["y1"] = min(max(box["y1"], 0.0), 1.0)
    box["x2"] = min(max(box["x2"], 0.0), 1.0)
    box["y2"] = min(max(box["y2"], 0.0), 1.0)

    return box


def main():
    hef = HEF(str(MODEL_PATH))

    input_info = hef.get_input_vstream_infos()[0]
    height, width, _ = input_info.shape

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("USB camera not opened")

    smoothed = None
    last_seen_time = 0
    hold_seconds = 0.7

    fps_time = time.time()
    fps_count = 0
    fps = 0.0

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_group = target.configure(hef, configure_params)[0]

        input_params = InputVStreamParams.make(
            network_group,
            format_type=FormatType.UINT8,
        )

        output_params = OutputVStreamParams.make(
            network_group,
            format_type=FormatType.FLOAT32,
        )

        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            with network_group.activate():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Camera frame failed")
                        break

                    resized = cv2.resize(frame, (width, height))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    input_data = {
                        input_info.name: np.expand_dims(rgb, axis=0).astype(np.uint8)
                    }

                    results = infer_pipeline.infer(input_data)
                    output = list(results.values())[0]

                    detections = parse_detections(output, conf_threshold=0.60)
                    best = choose_best_person(detections)

                    now = time.time()

                    if best is not None:
                        accept_detection = True
                        
                        if smoothed is not None:
                            old_cx = (smoothed["x1"] + smoothed["x2"]) / 2
                            old_cy = (smoothed["y1"] + smoothed["y2"]) / 2
                            new_cx = (best["x1"] + best["x2"]) / 2
                            new_cy = (best["y1"] + best["y2"]) / 2

                            jump = abs(new_cx - old_cx) + abs(new_cy - old_cy)

                            if jump > 0.35:
                                accept_detection = False

                        if accept_detection:
                            smoothed = smooth_box(smoothed, best, alpha=0.08)
                            smoothed = clamp_box(smoothed)
                            last_seen_time = now

                    else:
                        if now - last_seen_time > hold_seconds:
                            smoothed = None

                    fps_count += 1
                    if now - fps_time >= 1.0:
                        fps = fps_count / (now - fps_time)
                        fps_count = 0
                        fps_time = now

                    if smoothed is not None:
                        h, w = frame.shape[:2]
                        x1 = int(smoothed["x1"] * w)
                        y1 = int(smoothed["y1"] * h)
                        x2 = int(smoothed["x2"] * w)
                        y2 = int(smoothed["y2"] * h)

                        cx = (smoothed["x1"] + smoothed["x2"]) / 2
                        cy = (smoothed["y1"] + smoothed["y2"]) / 2
                        area = smoothed["area"]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (int(cx * w), int(cy * h)), 5, (0, 255, 0), -1)

                        label = f"PERSON {smoothed['score']:.2f} | cx:{cx:.2f} area:{area:.2f}"
                        cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "NO PERSON", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    cv2.imshow("Hailo Smooth Person Detection", frame)

                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
