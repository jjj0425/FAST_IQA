from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

# Load the uploaded image
image_path = "C:\\Users\\JJJ\\my\\data\\.images\\FAST\\rawdata\\RUQ\\segmentation\\img_256\\0412.png"
image = Image.open(image_path)

# Perform horizontal flip
flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

# Rotate the image by 15 degrees
rotated_image = image.rotate(15, resample=Image.BICUBIC, expand=True)

# Apply Gaussian blur
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))

# Save the processed images
flipped_image_path = "C:\\Users\\JJJ\\my\\data\\.images\\FAST\\rawdata\\RUQ\\flipped_0412.png"
rotated_image_path = "C:\\Users\\JJJ\\my\\data\\.images\\FAST\\rawdata\\RUQ\\rotated_0412.png"
blurred_image_path = "C:\\Users\\JJJ\\my\\data\\.images\\FAST\\rawdata\\RUQ\\blurred_0412.png"

flipped_image.save(flipped_image_path)
rotated_image.save(rotated_image_path)
blurred_image.save(blurred_image_path)

# Display the original and processed images for comparison
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(np.array(image))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Flipped Image")
plt.imshow(np.array(flipped_image))
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Rotated Image")
plt.imshow(np.array(rotated_image))
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Blurred Image")
plt.imshow(np.array(blurred_image))
plt.axis("off")

plt.show()
