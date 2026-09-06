import numpy as np
from PIL import Image
from scipy import ndimage

img = Image.open('me.jpg')
arr = np.array(img)
# The wall background is bright (R > 215, G > 210, B > 200)
# Let's check connected components starting from border
is_bg_color = (arr[:, :, 0] > 218) & (arr[:, :, 1] > 212) & (arr[:, :, 2] > 205)

# Label connected components of background
labeled, num_features = ndimage.label(is_bg_color)

# Find labels that touch the top border (definitely background)
top_labels = set(labeled[0, :])
left_labels = set(labeled[:500, 0])
right_labels = set(labeled[:500, -1])
bg_labels = (top_labels | left_labels | right_labels) - {0}

bg_mask = np.isin(labeled, list(bg_labels))

# Now subject mask is NOT background
subject_mask = ~bg_mask

# Fill any small holes inside the subject (e.g. if part of shirt collar was white)
subject_mask = ndimage.binary_fill_holes(subject_mask)

# Slight morphological smoothing
# Binary closing to smooth hair edges
subject_mask = ndimage.binary_closing(subject_mask, structure=np.ones((5, 5)))

# Convert to RGBA
alpha = (subject_mask.astype(np.uint8) * 255)
# Smooth alpha boundary slightly with Gaussian filter
alpha_smooth = ndimage.gaussian_filter(alpha.astype(float), sigma=1.2)
alpha_smooth = np.clip(alpha_smooth, 0, 255).astype(np.uint8)

rgba = np.dstack((arr, alpha_smooth))
out_img = Image.fromarray(rgba, 'RGBA')
out_img.save('me.png')
print("Successfully generated me.png with transparent background! Size:", out_img.size)
