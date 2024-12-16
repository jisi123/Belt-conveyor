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

"""Defines routines to compute mel spectrogram features from audio waveform."""
#定义从音频波形计算 mel 频谱图特征的例程

import numpy as np


def frame(data, window_length, hop_length):
  """Convert array into a sequence of successive possibly overlapping frames.
将数组转换为连续的可能重叠的帧序列。
  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.
形状 （num_samples， ...） 的 n 维数组转换为形状 （num_frames， window_length， ...） 的 （n+1）-D 数组，
其中每一帧从前一帧之后的 hop_length 点开始。
  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.
这是使用 stride_tricks 完成的，因此不会复制原始数据。但是，没有零填充，因此不包括末尾的任何不完整帧。
  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.
data： np.array 的维度为 N >= 1。window_length：每帧中的样本数。hop_length：在每个窗口之间前进（以样本为单位）。
  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.（N+1）-D np.array，其行数与可提取的完整帧数相同。
  """
  num_samples = data.shape[0]#：获取数据 data 的第一维（通常是时间维度）的长度，即样本数。
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
  #计算可以覆盖整个数据的帧数。这里，window_length 是窗口的长度，hop_length 是帧之间的跳跃长度。
  # np.floor 函数用于向下取整，确保帧数是整数。
  shape = (num_frames, window_length) + data.shape[1:]#构造一个元组 shape，它定义了新数组的形状。
  # num_frames 和 window_length 是第一维和第二维的大小，其余维度的大小与原始数据 data 的相应维度相同。
  strides = (data.strides[0] * hop_length,) + data.strides#构造一个元组 strides，它定义了新数组的步长。
  # 步长是数组中连续元素之间的字节数。这里，第一维的步长是原始数据第一维步长乘以跳跃长度，其余维度的步长与原始数据相同。
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
#使用 numpy 的 as_strided 函数创建一个新的视图数组。这个新数组的形状由 shape 指定，步长由 strides 指定。
# 这个函数允许在不复制数据的情况下创建新的数组视图。

def periodic_hann(window_length):
  """Calculate a "periodic" Hann window.

  The classic Hann window is defined as a raised cosine that starts and
  ends on zero, and where every value appears twice, except the middle
  point for an odd-length window.  Matlab calls this a "symmetric" window
  and np.hanning() returns it.  However, for Fourier analysis, this
  actually represents just over one cycle of a period N-1 cosine, and
  thus is not compactly expressed on a length-N Fourier basis.  Instead,
  it's better to use a raised cosine that ends just before the final
  zero value - i.e. a complete cycle of a period-N cosine.  Matlab
  calls this a "periodic" window. This routine calculates it.经典的 Hann 窗口被定义为一个凸余弦，
  它以零开始和结束，每个值出现两次，但奇数长度窗口的中点除外。Matlab 称其为“对称”窗口，np.hanning（） 返回它。
  然而，对于傅里叶分析，这实际上代表了周期 N-1 余弦的一个周期，因此不能在长度 N 傅里叶的基础上紧凑地表示。
  相反，最好使用在最终零值之前结束的升余弦 - 即周期 N 余弦的完整循环。Matlab称其为“周期性”窗口。此例程计算它。

  Args:
    window_length: The number of points in the returned window.返回窗口中的点数。

  Returns:
    A 1D np.array containing the periodic hann window.
  """
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  """Calculate the short-time Fourier transform magnitude.
计算短时傅里叶变换幅度
  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.
signal：输入时域信号的 1D np.array。fft_length：要应用的 FFT 的大小。
hop_length：在传递给 FFT 的每个帧之间前进（以样本为单位）。
window_length：要传递给 FFT 的每个样本块的长度。
  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
    其中每行包含相应输入样本帧的 FFT 的 fft_length2+1 唯一值的大小。
  """
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
  """Convert frequencies to mel scale using HTK formula.
使用 HTK 公式将频率转换为梅尔标度
  Args:
    frequencies_hertz: Scalar or np.array of frequencies in hertz.

  Returns:
    Object of same size as frequencies_hertz containing corresponding values
    on the mel scale.与frequencies_hertz大小相同，包含梅尔刻度上的相应值的对象。
  """
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
  """Return a matrix that can post-multiply spectrogram rows to make mel.
返回一个矩阵，该矩阵可以对频谱图行进行后乘以使 mel
  Returns a np.array matrix A that can be used to post-multiply a matrix S of
  spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
  "mel spectrogram" M of frames x num_mel_bins.  M = S A.
返回一个 np.array 矩阵 A，该矩阵可用于将频谱图值（STFT 幅度）的矩阵 S 后乘以帧 x bin，以生成帧 x num_mel_bins的“梅尔频谱图”M 。
M = S A。
  The classic HTK algorithm exploits the complementarity of adjacent mel bands
  to multiply each FFT bin by only one mel weight, then add it, with positive
  and negative signs, to the two adjacent mel bands to which that bin
  contributes.  Here, by expressing this operation as a matrix multiply, we go
  from num_fft multiplies per frame (plus around 2*num_fft adds) to around
  num_fft^2 multiplies and adds.  However, because these are all presumably
  accomplished in a single call to np.dot(), it's not clear which approach is
  faster in Python.  The matrix multiplication has the attraction of being more
  general and flexible, and much easier to read.
经典的 HTK 算法利用相邻 mel 条带的互补性，将每个 FFT bin 仅乘以一个 mel 权重，
然后将其与该条柱贡献的两个相邻 mel 条带相加（带有正号和负号）。在这里，通过将此操作表示为矩阵乘法，
我们从每帧 num_fft 次乘法（加上大约 2num_fft 次加法）到大约 num_fft^2 次乘法和加法。
但是，由于这些方法都是在对 np.dot（） 的单个调用中完成的，因此尚不清楚哪种方法在 Python 中更快。
矩阵乘法的吸引力在于更通用、更灵活，更易于阅读
  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.  This is
      the number of columns in the output matrix.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    audio_sample_rate: Samples per second of the audio at the input to the
      spectrogram. We need this to figure out the actual frequencies for
      each spectrogram bin, which dictates how they are mapped into mel.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum.  This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.
num_mel_bins：生成的梅尔光谱中有多少条带。这是输出矩阵中的列数。num_spectrogram_bins：源频谱图数据中有多少个箱，理解为fft_size2 + 1，即频谱图只包含非冗余的FFT箱。audio_sample_rate：频谱图输入处每秒采样的音频。
我们需要它来计算每个频谱图箱的实际频率，这决定了它们如何映射到 mel 中。
lower_edge_hertz：要包含在 mel 频谱中的频率的下限。这对应于最低三角形带的下边缘。upper_edge_hertz：最高频段的所需顶部边缘。
  Returns:
    An np.array with shape (num_spectrogram_bins, num_mel_bins).

  Raises:
    ValueError: if frequency edges are incorrectly ordered or out of range.
  """
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz < 0.0:
    raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > nyquist_hertz:
    raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
  # The i'th mel band (starting from i=1) has center frequency
  # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
  # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
  # the band_edges_mel arrays.
  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the *mel* domain, not hertz.
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  # HTK excludes the spectrogram DC bin; make sure it always gets a zero
  # coefficient.
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix


def log_mel_spectrogram(data,
                        audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):
  """Convert waveform to a log magnitude mel-frequency spectrogram.

  Args:
    data: 1D np.array of waveform data.
    audio_sample_rate: The sampling rate of data.
    log_offset: Add this to values when taking log to avoid -Infs.
    window_length_secs: Duration of each window to analyze.
    hop_length_secs: Advance between successive analysis windows.
    **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.
data：波形数据的 1D np.array。audio_sample_rate：数据的采样率。log_offset：在获取日志时将其添加到值中，以避免 -Infs。
 window_length_secs：要分析的每个窗口的持续时间。hop_length_secs：在连续的分析窗口之间前进。
kwargs：要传递给 spectrogram_to_mel_matrix 的其他参数。
  Returns:
    2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
    magnitudes for successive frames.
  """
  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)
  mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
      num_spectrogram_bins=spectrogram.shape[1],
      audio_sample_rate=audio_sample_rate, **kwargs))
  return np.log(mel_spectrogram + log_offset)
