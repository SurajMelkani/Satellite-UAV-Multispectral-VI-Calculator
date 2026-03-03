# Satellite & UAV Multispectral Vegetation Index Calculator

An interactive, web-based tool built with **Python** and **Streamlit** to rapidly process and visualize multispectral imagery and RGB photos from satellite and drone platforms.

**[Launch Live Satellite & UAV Multispectral Vegetation Index Calculator]**(https://satellite-uav-multispectral-vi-calculator.streamlit.app/]
## Overview

This application allows users to upload different, individual band GeoTIFFs (such as Red, Green, NIR, Red Edge, and Blue) or a standard RGB photo to dynamically compute up to 11 distinct vegetation indices. It processes the imagery at a preview resolution for rapid rendering and requires no Coordinate Reference System (CRS) transformations, making it highly efficient for quick field assessments.

📁 **Demo Data Included**
To test the platform, a complete set of sample multispectral bands and an RGB photo are available in the `sample_data/` folder of this repository. Simply download them and upload them into the live platform.

### Supported Indices:
* **NDVI, GNDVI, NDRE, SAVI** (Standard & Soil-Adjusted)
* **EVI, EVI2** (Enhanced Vegetation Indices for high biomass)
* **NGRDI, RENDVI, NIRRENDVI** (Red-Edge and Visible contrast)
* **RGBVI, VARI** (RGB-only indices)

---

## How to Run Locally

To run this application on your own machine, clone this repository and run the app through your bash or command prompt after installing the dependencies listed in the requirements.txt file.
   
