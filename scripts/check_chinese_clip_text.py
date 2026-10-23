import numpy as np
import torch

from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from photoindex.retrieval import encode_texts


MODEL_DIR = "local-data/models/chinese-clip-base"

text = "夜晚沿水边延伸的一长排暖黄色灯笼"


def main():
    processor = ChineseCLIPProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )

    model = ChineseCLIPModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model.eval()

    # A. 使用我们修复后的独立文本编码接口
    our_vector = encode_texts(
        texts=[text],
        model=model,
        processor=processor,
        device="cpu",
    )

    # B. 使用官方完整模型 forward 作为参考
    synthetic_image = Image.new(
        "RGB",
        (224, 224),
        color="white",
    )

    inputs = processor(
        text=[text],
        images=synthetic_image,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    official_vector = (
        outputs.text_embeds
        .float()
        .cpu()
        .numpy()
    )

    print("Our shape:", our_vector.shape)
    print("Official shape:", official_vector.shape)

    # 检查形状、有限数值和数值一致性
    assert our_vector.shape == (1, 512)
    assert official_vector.shape == (1, 512)

    assert np.isfinite(our_vector).all()
    assert np.isfinite(official_vector).all()

    np.testing.assert_allclose(
        our_vector,
        official_vector,
        rtol=1e-4,
        atol=1e-5,
    )

    print("Text embedding matches model.forward()")
    print("Regression check passed")


if __name__ == "__main__":
    main()