# Focus Stacking Application

This project implements a robust focus stacking pipeline that fuses multiple images taken at different focal depths into a single all-in-focus image. It utilizes Laplacian Pyramid fusion with sharpness-based decision masks and includes an ECC-based image alignment step.

## Features

*   **Advanced Fusion Algorithm**: Uses Laplacian Pyramids and local energy maps for high-quality fusion.
*   **Image Alignment**: Automatically aligns source images using the ECC algorithm to correct for minor camera movements.
*   **Performance Optimization**: Caches aligned images to significantly speed up subsequent runs.
*   **Interactive GUI**: A user-friendly graphical interface to select datasets, adjust parameters, and visualize results with source image animation.
*   **Configurable Parameters**: Adjust pyramid levels and mask types (Hard vs. Soft) to fine-tune results.

## Prerequisites

*   Python 3.8 or higher
*   Visual Studio Code (recommended)

## Installation

1.  **Clone or Download the Repository**
    Download the project folder to your local machine.

2.  **Open in VS Code**
    *   Launch Visual Studio Code.
    *   Go to **File** -> **Open Folder...**
    *   Select the project root folder.

3.  **Run Setup Script**
    We provide automated scripts to set up the virtual environment, install dependencies, and download necessary data.

    *   **Windows (PowerShell)**:
        ```powershell
        .\setup.ps1
        ```
        *Note: If you encounter a security error, try running: `PowerShell -ExecutionPolicy Bypass -File .\setup.ps1`*

    *   **macOS / Linux / Git Bash**:
        ```bash
        bash setup.sh
        ```

4.  **Activate Virtual Environment**
    After setup is complete, activate the environment to run the application.

    *   **Windows**:
        ```powershell
        .venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

## Usage

### Running the GUI

To start the application, run `gui.py`:

```bash
python gui.py
```
The Graphical User Interface provides the easiest way to use the tool.
*   Select an image set from the dropdown.
*   Choose your preferred mask type and pyramid levels.
*   Click **Generate Fused Image**.

### Running the Command Line Script
For batch processing or debugging, you can use the main script directly.

```bash
cd core
python main.py
```

## Project Structure

*   `core/`: Contains the source code for the fusion algorithm and GUI.
*   `data/`: Directory for input image datasets.
*   `output/`: Generated results are saved here.
