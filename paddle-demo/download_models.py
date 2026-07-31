"""Pre-download PP-StructureV3 weights at image build time.

Baking the models into the image (rather than fetching ~1 GB on first request)
is what keeps ACA replica cold starts reasonable.

Downloading does not require a GPU, so this always builds the pipeline on CPU;
the runtime device is chosen separately via PADDLE_DEVICE.
"""
import numpy as np
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    lang="en",
    layout_detection_model_name="PP-DocLayout_plus-L",
    use_table_recognition=False,
    use_chart_recognition=False,
    use_formula_recognition=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)

# One dummy pass forces every sub-model to be fetched and its graph built.
list(pipeline.predict(np.full((640, 480, 3), 255, dtype="uint8")))
print("PP-StructureV3 weights cached")
