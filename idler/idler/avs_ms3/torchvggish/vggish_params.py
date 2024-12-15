# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Global parameters for the VGGish model.

See vggish_slim.py for more information.
"""

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.定义输入mel频谱图片段的帧数为96。
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.定义输入mel频谱图片段的频带数为64。
EMBEDDING_SIZE = 128  # Size of embedding layer.定义嵌入层的大小为128。

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000 #定义音频的采样率为16000赫兹
STFT_WINDOW_LENGTH_SECONDS = 0.025 #定义短时傅里叶变换（STFT）窗口的长度为0.025秒。
STFT_HOP_LENGTH_SECONDS = 0.010 #定义STFT的跳跃长度为0.010秒。
NUM_MEL_BINS = NUM_BANDS #定义mel滤波器组的bin数，与频带数相同。
MEL_MIN_HZ = 125  #定义mel频谱的最小频率为125赫兹。
MEL_MAX_HZ = 7500#定义mel频谱的最大频率为75000赫兹。
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.定义输入mel频谱图的稳定对数偏移量
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames定义每个样本包含的帧的窗口大小为0.96秒
EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.定义样本之间的跳跃长度为0.96秒，这里没有重叠

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'#定义PCA特征向量的名称
PCA_MEANS_NAME = 'pca_means'#定义PCA均值的名称。
QUANTIZE_MIN_VAL = -2.0#定义量化的最小值为-2.0
QUANTIZE_MAX_VAL = +2.0#定义量化的最大值为+2.0。

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.定义权重初始化时使用的标准差为0.01。
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.定义Adam优化器的学习率为0.0001。
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.定义Adam优化器的epsilon参数为0.00000001。

# Names of ops, tensors, and features.
INPUT_OP_NAME = 'vggish/input_features' #定义输入操作的名称。
INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0' #定义输入张量的名称，这里使用了TensorFlow的命名约定。
OUTPUT_OP_NAME = 'vggish/embedding'   #定义输出操作的名称。
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'   #定义输出张量的名称。
AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'  #定义音频嵌入特征的名称。
