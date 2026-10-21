import time
from chinese_clip_checkpoint import MODEL_DIR, LOCK_FILE, check_revision, require_model_files


def main():
    # Check before importing AI libraries, so an unfinished download produces
    # a useful error instead of a Transformers checkpoint-loading traceback.
    try:
        check_revision(MODEL_DIR, LOCK_FILE)
        require_model_files(MODEL_DIR)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    import torch
    from PIL import Image, ImageDraw
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    # 1. 生成合成图片，不读取私人图库
    image = Image.new(
        "RGB",
        (900, 900),
        color=(242, 240, 233),
    )

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        (180, 120, 720, 780),
        fill=(178, 58, 64),
    )

    draw.ellipse(
        (330, 300, 570, 540),
        fill=(233, 192, 111),
    )

    text = "红色矩形中有一个黄色圆形"

    # 2. 固定第一次实验设备
    device = "cpu"

    # 3. 读取模型与 Processor
    start = time.perf_counter()

    processor = ChineseCLIPProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )

    model = ChineseCLIPModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    loading_seconds = time.perf_counter() - start

    # 4. 预处理图片
    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    # 5. 预处理中文
    text_inputs = processor(
        text=[text],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    # 6. 检查文本没有超出模型限制
    text_length = text_inputs["input_ids"].shape[1]
    max_length = model.config.text_config.max_position_embeddings

    if text_length > max_length:
        raise ValueError("Text exceeds model position limit")

    # 7. 将输入 Tensor 移动到执行设备
    image_inputs = {
        key: value.to(device)
        for key, value in image_inputs.items()
    }

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    # 8. 推理，不计算梯度
    with torch.inference_mode():
        # Transformers 4.57.1 get_text_features() incorrectly uses
        # pooler_output although this model disables the text pooler.
        # Its joint forward correctly projects the final CLS hidden state.
        outputs = model(
            **image_inputs,
            **text_inputs,
            return_dict=True,
        )
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # 检查返回类型
        if not torch.is_tensor(image_features):
            raise TypeError("Unexpected image feature type")

        if not torch.is_tensor(text_features):
            raise TypeError("Unexpected text feature type")

        # 统一 float32
        image_features = image_features.float()
        text_features = text_features.float()

        # 归一化前先检查数值
        for features in (image_features, text_features):
            if not torch.isfinite(features).all():
                raise ValueError("Non-finite embedding")

            if torch.any(features.norm(dim=-1) == 0):
                raise ValueError("Zero embedding")

        # L2 归一化
        image_vector = torch.nn.functional.normalize(
            image_features, dim=-1
        )

        text_vector = torch.nn.functional.normalize(
            text_features, dim=-1
        )

        # 两个单位向量的点积
        score = image_vector @ text_vector.T

    # 9. 运行后验收
    expected = (1, model.config.projection_dim)

    assert image_vector.shape == expected
    assert text_vector.shape == expected

    assert torch.isfinite(score).all()

    print("Device:", device)
    print("Model load seconds:", loading_seconds)
    print("Image shape:", tuple(image_vector.shape))
    print("Text shape:", tuple(text_vector.shape))
    print("Image norm:", image_vector.norm().item())
    print("Text norm:", text_vector.norm().item())
    print("Similarity:", score.item())
    print("Smoke test passed")


if __name__ == "__main__":
    main()
