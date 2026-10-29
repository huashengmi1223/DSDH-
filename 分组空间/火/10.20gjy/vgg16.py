# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VGG16 公共接口，使用已安装的 Keras 官方实现。"""
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

__all__ = ["VGG16", "preprocess_input", "decode_predictions"]

if __name__ == "__main__":
    import tensorflow as tf
    model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    print("VGG16 前向运行成功：", model(tf.zeros((1, 32, 32, 3))).shape)
