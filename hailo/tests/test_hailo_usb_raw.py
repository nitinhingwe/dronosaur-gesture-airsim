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


def main():
    print(f"Loading model: {MODEL_PATH}")

    hef = HEF(str(MODEL_PATH))

    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]

    print("Input:", input_info)
    print("Output:", output_info)

    input_shape = input_info.shape
    print("Input shape:", input_shape)

    # Usually shape is [height, width, channels]
    height = input_shape[0]
    width = input_shape[1]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("USB camera not opened")

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]

        input_vstreams_params = InputVStreamParams.make(
            network_group,
            format_type=FormatType.UINT8,
        )

        output_vstreams_params = OutputVStreamParams.make(
            network_group,
            format_type=FormatType.FLOAT32,
        )

        with InferVStreams(
            network_group,
            input_vstreams_params,
            output_vstreams_params,
        ) as infer_pipeline:

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

                    print("\nResult keys:", results.keys())
                    for key, value in results.items():
                        print("Output:", key)
                        print("Type:", type(value))
                        try:
                            print("Shape:", value.shape)
                            print("Sample:", value.flatten()[:20])
                        except Exception:
                            print("Value:", value)

                    cv2.imshow("USB Camera Raw Hailo Test", frame)

                    # Press ESC to quit
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
