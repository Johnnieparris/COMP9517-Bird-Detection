# Video script: advanced studies, discussion and future work (Member E's share, presented by Johnnie)

Target: about 1:45 spoken (~240 words at a steady pace). Slide cues in brackets; slides are
outlined in `member_e_slides.md`. Every number is on the **test** split and comes from a
committed CSV: `results/robustness_test.csv`, `results/occluder_shortcut_val.csv`,
`results/data_efficiency_test.csv`, `results/failure_categories_test.csv`, `analysis/size_bins.csv`.

---

**[Slide 1: robustness, top row of e_robustness.png. Webcam on.]**

I'm Johnnie Parris again. I'm presenting the advanced studies, which were Member E's share of
the project. First: how robust are our detectors? We corrupted the test images five ways, at
three strengths. The CNNs stay far ahead, but they're fragile: on average they keep only about
63 percent of their score, and occlusion hurts every method most. Faster R-CNN copes with heavy
noise much better than YOLO.

**[Slide 2: bottom row, robust training]**

Can we fix that? Denoising before the classical methods gave no real gain. Training YOLO on
corrupted images worked: clean accuracy stayed at 0.875, and the average over corrupted images
rose from 0.56 to 0.77. But under occlusion it scored even higher than on clean images. A
quick test explained why: it had learned that a grey patch means "bird", and fired on 96
percent of grey squares we pasted onto empty sand.

**[Slide 3: e_data_efficiency.png]**

How much data does YOLO need? Half the training set scores 0.874, almost the full set's 0.876.
Deleting a fifth of the boxes did no harm, but jittering box edges by 20 percent cost three
points. Stronger augmentation at 10 percent of the data gave no clear gain.

**[Slide 4: e_failures.png and the flock overlay]**

Where do the detectors still fail? Mostly on birds inside flocks, and on small birds: YOLO finds
89 percent of large birds but only 61 percent of small ones. Next steps: tiled high-resolution
inference, density-based counting for flocks, and occlusion augmentation that isn't tied to the
labels.
