"""篆体字 VGG16 迁移学习。默认完整训练；--smoke-test 快速验证全流程。"""
import argparse
import os
import json
from pathlib import Path
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent.parent / ".keras"))
import tensorflow as tf
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.losses import BinaryCrossentropy
from keras.optimizers.legacy import Adam
from vgg16 import VGG16, preprocess_input

BASE_DIR = Path(__file__).resolve().parent


def add_input_top_model(base_model, class_num, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    # 预处理只在模型内执行一次，生成器保持原始 0~255 像素。
    x = base_model(preprocess_input(inputs))
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1)(x) if class_num == 2 else Dense(class_num, activation="softmax")(x)
    return Model(inputs, outputs)


def model_compile(model, learning_rate=0.001):
    binary = model.output_shape[-1] == 1
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=BinaryCrossentropy(from_logits=True) if binary else "categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)] if binary else ["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "Ret_Model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weights", default="imagenet", help="imagenet、权重文件路径或 none")
    parser.add_argument("--fine-tune", action="store_true", help="同时训练 VGG16 主干；默认冻结主干")
    parser.add_argument("--smoke-test", action="store_true", help="32 像素、2 张一批，训练/验证各一步，共一轮")
    parser.add_argument("--class-limit", type=int, default=None, help="仅使用前 N 个有效类别")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"没有数据目录 {args.data_dir}，请先运行 TTF2IMG.py 和 DeleteBlankIMG.py")
    # 每类至少 5 张，才能按 20% 划分出一张验证图片。
    classes = sorted(p.name for p in args.data_dir.iterdir()
                     if p.is_dir() and not p.name.startswith(".")
                     and sum(f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                             for f in p.iterdir()) >= 5)
    valid_count = len(classes)
    classes = classes[:args.class_limit]
    if len(classes) < 2:
        raise ValueError("至少需要两个类别，且每类至少 5 张图片，以划分训练集和验证集")
    if args.smoke_test:
        args.image_size, args.batch_size, args.epochs = 32, 2, 1
    tf.keras.utils.set_random_seed(42)
    shape = (args.image_size, args.image_size, 3)
    binary = len(classes) == 2
    gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    options = dict(directory=str(args.data_dir), classes=classes,
                   target_size=shape[:2], batch_size=args.batch_size,
                   class_mode="binary" if binary else "categorical", seed=42)
    train = gen.flow_from_directory(subset="training", **options)
    val = gen.flow_from_directory(subset="validation", shuffle=False, **options)
    weights = None if args.weights.lower() == "none" else args.weights
    cached_weights = BASE_DIR.parent / ".keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if weights == "imagenet" and cached_weights.is_file():
        import hashlib
        if hashlib.md5(cached_weights.read_bytes()).hexdigest() != "6d6bbae143d832006294945121d1f1fc":
            raise ValueError("本地 VGG16 权重校验失败，请重新下载")
        weights = str(cached_weights)
    base = VGG16(input_shape=shape, include_top=False, weights=weights)
    base.trainable = args.fine_tune
    model = model_compile(add_input_top_model(base, len(classes), shape))
    print(f"使用 {len(classes)} 类（满足数量要求的类别共 {valid_count}）；主干可训练：{base.trainable}", flush=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    history = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks,
                        steps_per_epoch=1 if args.smoke_test else None,
                        validation_steps=1 if args.smoke_test else None,
                        workers=1, use_multiprocessing=False, verbose=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    # 保存类别映射供预测结果解释使用。
    mapping_path = args.output.parent / (args.output.name + "-classes.json")
    mapping_path.write_text(json.dumps(train.class_indices, ensure_ascii=False, indent=2))
    loaded = tf.keras.models.load_model(str(args.output))
    images, _ = val[0]
    predictions = loaded(images[:1], training=False)
    tf.debugging.assert_all_finite(predictions, "预测结果包含非有限值")
    report = {"smoke_test": args.smoke_test, "classes": len(classes),
              "training_images": train.samples, "validation_images": val.samples,
              "epochs_completed": len(history.epoch), "image_size": args.image_size,
              "weights": args.weights, "history": history.history,
              "reload_prediction_shape": predictions.shape.as_list()}
    (args.output.parent / (args.output.name + "-run.json")).write_text(json.dumps(report, indent=2))
    print(f"训练、验证、模型保存和重新加载成功：{args.output}", flush=True)


if __name__ == "__main__":
    main()
