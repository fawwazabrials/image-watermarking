import numpy as np
import sys
from PIL import Image
import math
from scipy.signal import convolve2d
import seaborn as sns
import matplotlib.pyplot as plt

class ImageWatermarking:
    def __init__(self, original_image_name, seed, scaling_factor, save= True):
        self.original_image_path = f'sample/{original_image_name}.png'
        self.noisy_pattern_path = f'result/noisy-pattern-{seed}.png'
        self.watermarked_image_path = f'result/watermarked-{original_image_name}-{scaling_factor}.png'
        self.convolved_image_path = f'result/convolved-{original_image_name}-{scaling_factor}.png'
        # self.convolved_image_path = f'result/convolved-{original_image_name}.png'

        self.scaling_factor = scaling_factor
        self.seed = seed
        self.save = save
        
        self.original_image = self.open_image(self.original_image_path)
        self.size = self.original_image.shape

        # print(self.original_image)

    def save_image(self, image_array, file_path):
        image = Image.fromarray((image_array).astype(np.uint8), 'L')
        image.save(file_path)

    def open_image(self, file_path):
        original_image = Image.open(file_path).convert('L')
        original_array = np.array(original_image)
        return original_array
    
    def normalize(self, pattern):
        pattern = (pattern - np.mean(pattern)) / np.std(pattern)
        return pattern

    def generate_noisy_pattern(self, seed=None):
        if seed is None: seed = self.seed
        Generator = np.random.default_rng(seed)
        noisy_pattern = Generator.integers(0, 2, size=self.size)
        noisy_pattern = 2 * noisy_pattern - 1

        if (self.save):
            self.save_image(noisy_pattern, self.noisy_pattern_path)

        self.noisy_pattern = noisy_pattern
        return noisy_pattern
    
    def generate_watermarked_image(self):
        watermark_pattern = self.generate_noisy_pattern()
        watermark_pattern = watermark_pattern[:self.original_image.shape[0], :self.original_image.shape[1]]
        watermarked_image = self.original_image + self.scaling_factor * watermark_pattern
        watermarked_image = np.clip(watermarked_image, 0, 255)

        if (self.save):
            self.save_image(watermarked_image, self.watermarked_image_path)

        self.watermarked_image = watermarked_image
        return watermarked_image
    
    def detect_watermark(self, seed=None):
        noisy_pattern = self.generate_noisy_pattern(seed=seed)
        normalized_noisy_array = (noisy_pattern - np.mean(noisy_pattern)) / np.std(noisy_pattern)
        edge_enhanced_image = convolve2d(self.watermarked_image, np.divide(np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]), 2), mode='same')
        normalized_edge_enhanced_image = (edge_enhanced_image - np.mean(edge_enhanced_image)) / np.std(edge_enhanced_image)

        if (self.save):
            self.save_image(edge_enhanced_image, self.convolved_image_path)
        corr = np.mean(normalized_edge_enhanced_image * normalized_noisy_array)
        return corr

gain_factor = 25
imageWatermarking = ImageWatermarking('eva', 15012003, gain_factor)
imageWatermarking.generate_watermarked_image()
corr = imageWatermarking.detect_watermark()
print(f'k={gain_factor} :',corr)

# for i in range(11):
#     imageWatermarking = ImageWatermarking('dog', 15012003, math.ceil((1.63238*i*i + 9.20284*i - 0.259206)/5)*5)
#     imageWatermarking.generate_watermarked_image()
#     corr = imageWatermarking.detect_watermark()
#     print(f'k={math.ceil((1.63238*i*i + 9.20284*i - 0.259206)/5)*5} :',corr)

# x = []
# y = []

# for i in range(256):
#     imageWatermarking = ImageWatermarking('eva', 15012003, i, save=False)
#     imageWatermarking.generate_watermarked_image()
#     corr = imageWatermarking.detect_watermark(seed=1)
#     print(f'k={i} :',corr)
#     x.append(i)
#     y.append(corr)


# print(x)
# print(y)

# sns.set_style("darkgrid")
# plt.plot(y)
# plt.savefig("eva_wrong_plot.png")



# ImageWatermarking = ImageWatermarking('dog', 15012003, 10)


