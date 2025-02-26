﻿# **CSV Data Visualizer**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Dash](https://img.shields.io/badge/Dash-2.0%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-red)

A **web-based application** built with Python and Dash that allows users to upload CSV files, visualize data, perform statistical analysis, and edit data in real-time.

---

## **Features**

- **Upload CSV Files**: Easily upload and parse CSV files.
- **Data Visualization**: Visualize data using various plot types:
  - Scatter Plot
  - Line Plot
  - Bar Chart
  - Histogram
  - Box Plot
  - Violin Plot
  - Density Plot
  - Correlation Heatmap
  - Pair Plot
  - 3D Scatter Plot
- **Data Editing**: Edit data directly in the table.
- **Statistical Analysis**:
  - View summary statistics (mean, median, mode, etc.) for the entire dataset.
  - Calculate statistics for specific columns.
- **Download Edited Data**: Download the edited CSV file.
  
---

## **How to Use**

1. **Upload a CSV File**:
   - Click the "Drag and Drop or Select a CSV File" button to upload your CSV file.
2. **View Data**:
   - The first 5 rows of the data will be displayed in a table.
3. **Visualize Data**:
   - Select a plot type from the dropdown menu to visualize the data.
4. **Edit Data**:
   - Edit the data directly in the table.
5. **Calculate Statistics**:
   - View summary statistics for the entire dataset or specific columns.
6. **Download Edited Data**:
   - Click the "Download Edited CSV" button to download the updated CSV file.

---

## **Installation**

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/csv-data-visualizer.git
   cd csv-data-visualizer
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```
4. Open your browser and go to `http://127.0.0.1:8050/`.
