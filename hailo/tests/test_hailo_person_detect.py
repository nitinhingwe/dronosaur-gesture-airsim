import cv2
import numpy as np
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


def parse_detections(output, conf_threshold=0.4):
    detections = []

    def walk(obj, class_id=0):
        # If obj is a list/tuple, go deeper
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                walk(item, idx)
            return

        # If obj is numpy array
        arr = np.asarray(obj)

        if arr.size == 0:
            return

        # Case: one detection row like [x1, y1, x2, y2, score]
        if arr.ndim == 1 and arr.shape[0] >= 5:
            x1, y1, x2, y2, score = arr[:5]

            if float(score) >= conf_threshold:
                detections.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score),
                    "class": int(class_id),
                })
            return

        # Case: multiple rows
        if arr.ndim >= 2 and arr.shape[-1] >= 5:
            arr = arr.reshape(-1, arr.shape[-1])

            for det in arr:
                x1, y1, x2, y2, score = det[:5]

                if float(score) >= conf_threshold:
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


def main():
    hef = HEF(str(MODEL_PATH))

    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]

    height, width, _ = input_info.shape

    cap = cv2.VideoCapture(0)

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
                        break

                    resized = cv2.resize(frame, (width, height))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    input_data = {
                        input_info.name: np.expand_dims(rgb, axis=0).astype(np.uint8)
                    }

                    results = infer_pipeline.infer(input_data)

                    output = list(results.values())[0]

                    detections = parse_detections(output)

                    for det in detections:
                        x1 = int(det["x1"] * frame.shape[1])
                        y1 = int(det["y1"] * frame.shape[0])
                        x2 = int(det["x2"] * frame.shape[1])
                        y2 = int(det["y2"] * frame.shape[0])

                        score = det["score"]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow("Hailo Person Detection", frame)

                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
