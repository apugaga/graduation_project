import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Paths to your data files
img_path = r"F:\laboratory\Graduation_thesis\data\data_fish\output_similar\sub-ACN0004_DTIFCT_weighted_similar.nii.gz"
atlas_path = r"F:\laboratory\Graduation_thesis\data\mask\JHU-ICBM-labels-2mm.nii"
xml_path = r"F:\laboratory\Graduation_thesis\data\mask\JHU-labels_位置資訊.xml"

# Load the subject image and JHU atlas
img = nib.load(img_path).get_fdata()
atlas = nib.load(atlas_path).get_fdata()

# Parse the XML to build a mapping from region name to label index
tree = ET.parse(xml_path)
root = tree.getroot()
label_map = {}
for lbl in root.find('data').findall('label'):
    name = lbl.text.strip()
    idx = int(lbl.get('index'))
    label_map[name] = idx

# Print out the label_map information
print("Loaded label_map:")
for region, idx in label_map.items():
    print(f"  {region!r}: index {idx}")

# Define the regions of interest (including original and newly added)
regions = [
    "Posterior corona radiata R",
    "Retrolenticular part of internal capsule R",
    "Anterior corona radiata R",
    "Tapetum L",
    "Superior corona radiata R",
    "Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) L",
    "Genu of corpus callosum",
    "Anterior corona radiata L",
    "Body of corpus callosum",
    "Posterior corona radiata L"
]

# Process each region: find best z-slice and overlay
for region_name in regions:
    if region_name not in label_map:
        print(f"Warning: '{region_name}' not found in XML labels.")
        continue
    label_val = label_map[region_name]
    region_mask = (atlas == label_val)
    # Count voxels per z-slice
    counts_per_z = region_mask.sum(axis=(0,1))
    best_z = np.argmax(counts_per_z)
    voxel_count = counts_per_z[best_z]
    print(f"{region_name} (index {label_val}) -> best z-slice: {best_z}, voxels: {voxel_count}")

    # Extract the corresponding 2D slices
    slice_img = img[:, :, best_z]
    slice_mask = region_mask[:, :, best_z]

    # Plot with overlay
    plt.figure(figsize=(6,6))
    plt.imshow(slice_img.T, cmap='gray', origin='lower')
    plt.imshow(
        np.ma.masked_where(~slice_mask.T, slice_mask.T),
        cmap='Reds', alpha=0.5, origin='lower'
    )
    plt.title(f"{region_name}: z={best_z}")
    plt.axis('off')

plt.show()
