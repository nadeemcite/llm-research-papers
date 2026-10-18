---
title: "YOLO: Talk & Q&A"
parent: "032 YOLO"
nav_order: 2
---
# TALK — You Only Look Once (YOLO)

## Talks, Slides and Official Pages

1. **CVPR 2016 talk by Joseph Redmon** — Redmon presented YOLO at CVPR 2016. The recording is on the Computer Vision Foundation YouTube channel: [You Only Look Once: Unified, Real-Time Object Detection](https://www.youtube.com/watch?v=NM6lrxy0bxs). The [CVPR 2016 slides](https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit?usp=sharing) are linked from Redmon's [YOLO publication page](https://pjreddie.com/publications/yolo/).

2. **OpenCV People's Choice Award** — The paper won the OpenCV People's Choice Award at CVPR 2016, as listed on [pjreddie.com](https://pjreddie.com/publications/yolo/). The same page also shares the paper's peer reviews.

3. **TED2017 talk** — Redmon explained real-time object detection to a general audience in his TED talk [How computers learn to recognize objects instantly](https://www.ted.com/talks/joseph_redmon_how_computers_learn_to_recognize_objects_instantly), with a live demo.

4. **Darknet and the YOLO project page** — The code was released as open source in the [Darknet framework](https://github.com/pjreddie/darknet). The [Darknet YOLO page](https://pjreddie.com/darknet/yolo/) shows how to run the later versions (YOLOv2/v3).

5. **Official paper pages** — [arXiv 1506.02640](https://arxiv.org/abs/1506.02640) and the [CVPR 2016 open-access version](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html).

## Q&A (answered from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/1506.02640).

**Q: Why frame detection as a regression problem instead of classification?**

From the paper (Abstract): "Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation."

**Q: YOLO makes more localization errors than R-CNN. Why?**

From the paper (Section 2.4, Limitations of YOLO): "YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict. Our model struggles with small objects that appear in groups, such as flocks of birds."

**Q: What motivated the unified, single-network approach?**

In plain words: older systems were a chain of separate parts (find regions, extract features, classify, clean up boxes), and each part was tuned on its own. YOLO puts all of these steps into one network. The paper says that because the whole pipeline is a single network, it can be trained end-to-end directly for detection quality, and that it combines these components "into a single, jointly optimized model" (Section 3).

**Q: How does YOLO compare to two-stage detectors in terms of false positives?**

From the paper (Abstract): "Compared to state-of-the-art detection systems, YOLO makes more localization errors but is less likely to predict false positives on background. Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when generalizing from natural images to other domains like artwork."

**Q: What was the speed advantage?**

From the paper (Abstract): "Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors."

## Common Misconceptions

1. **Misconception: "YOLOv1 uses anchor boxes."** — It does not. YOLOv1 predicts box coordinates directly from each grid cell. Anchor boxes arrived later, in YOLOv2 (YOLO9000).

2. **Misconception: "YOLO is both faster and more accurate than R-CNN-style detectors."** — YOLO is much faster, but in the original paper it is less accurate. On PASCAL VOC 2007, YOLO scores 63.4% mAP while Fast R-CNN scores 70.0%. YOLO's big win is speed, not top accuracy.

3. **Misconception: "'Only once' means no extra steps at all."** — The name means one forward pass of the network per image at test time. Training still takes many epochs, and non-maximal suppression is still used afterwards to remove duplicate boxes (it is a light step compared with region proposals).

4. **Misconception: "The loss function is just plain MSE."** — The base is sum-squared error, but the loss has several weighted parts: coordinate loss (λ_coord = 5), confidence loss for cells with and without objects (λ_noobj = 0.5), and classification loss. Width and height are predicted as square roots, so small boxes are not ignored.

5. **Misconception: "Each grid cell can detect several different objects."** — Each cell predicts B=2 boxes in the paper, but only one set of class probabilities, so in practice it can only label one class. The notebook simplifies further to B=1 box per cell.

## Real Citations

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *CVPR 2016*, pp. 779-788. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

2. Redmon, J., & Farhadi, A. (2017). "YOLO9000: Better, Faster, Stronger." *CVPR 2017*, pp. 7263-7271. (YOLOv2, extending v1 to 9000 classes) [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)

3. Liu, W., et al. (2016). "SSD: Single Shot MultiBox Detector." *ECCV 2016*. (Another single-stage detector from the same period) [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)

4. Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*. (RetinaNet, addressing class imbalance in single-stage detectors like YOLO) [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

5. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." *arXiv:1804.02767*. (Third generation, multi-scale predictions) [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)

**Citation count:** The original YOLO paper has about 53,000 citations on [OpenAlex](https://openalex.org/works/W1483870316) (CVPR 2016 version, checked October 2026), which makes it one of the most cited computer vision papers.
