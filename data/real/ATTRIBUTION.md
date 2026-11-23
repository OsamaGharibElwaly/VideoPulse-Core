# Real footage attribution

The real-video track uses four public sample clips from the
[Intel OpenVINO sample-videos](https://github.com/intel-iot-devkit/sample-videos) repository.
That repository licenses its material under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

| Local file | Upstream source | Used for |
|---|---|---|
| `videos/classroom.mp4` | [classroom.mp4](https://github.com/intel-iot-devkit/sample-videos/blob/master/classroom.mp4) | Real education scenario |
| `videos/worker-zone-detection.mp4` | [worker-zone-detection.mp4](https://github.com/intel-iot-devkit/sample-videos/blob/master/worker-zone-detection.mp4) | Real manufacturing / safety-zone scenario |
| `videos/store-aisle-detection.mp4` | [store-aisle-detection.mp4](https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4) | Real retail traffic scenario |
| `videos/people-detection.mp4` | [people-detection.mp4](https://github.com/intel-iot-devkit/sample-videos/blob/master/people-detection.mp4) | Real facility / pedestrian scenario |

The timestamped JSON side-cars in `logs/` are **generated locally** from YOLO26
detections on frames sampled from those clips by
[`tools/prepare_real_data.py`](../../tools/prepare_real_data.py). They are derived data,
not upstream annotations.

Model weights (`weights/yolo26n.pt`) come from the
[Ultralytics YOLO26 release assets](https://github.com/ultralytics/assets/releases).
Weights and videos are intentionally git-ignored because they are large binaries;
`python tools/prepare_real_data.py --download` recreates them locally.
