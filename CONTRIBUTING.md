# Contributing to Gunshot Detection on Edge Devices

Thank you for your interest in contributing! We are building an open-source solution to enhance national security and public safety through edge AI. Whether you are an AI researcher, FPGA engineer, or student, your contributions are welcome.

## 🤝 How to Contribute

### 1. Reporting Bugs
* Check the **Issues** tab to see if the bug is already reported.
* Open a new Issue using the "Bug Report" template.
* Include your hardware setup (e.g., Raspberry Pi 4, PYNQ-Z2) and logs.

### 2. Suggesting Enhancements
* We have a roadmap in the `Plans/` folder. Please check it first!
* If you have a new idea (e.g., "Thermal Camera Integration"), open an Issue with the tag `enhancement`.

### 3. Pull Requests (PRs)
1.  **Fork** the repository.
2.  **Clone** your fork locally.
3.  Create a branch: `git checkout -b feature/AmazingFeature`.
4.  **Commit** your changes: `git commit -m 'Add some AmazingFeature'`.
5.  **Push** to the branch: `git push origin feature/AmazingFeature`.
6.  Open a **Pull Request** and describe your changes.

## 🛠️ Development Guidelines

### 🐍 Python / AI Models
* **Code Style:** We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
* **Experiments:** If you train a new model, please place your notebook in `MFCC_YamNet/` or a new folder, and update the Results table in the README.
* **Dependencies:** Ensure `requirements.txt` is updated if you add new libraries.

### ⚡ FPGA / Hardware
* **Language:** Verilog / VHDL / HLS (C++).
* **Simulation:** All hardware modules must include a testbench.
* **Target:** Primary targets are Xilinx Zynq series (PYNQ, ZedBoard), but generic RTL is preferred.

## 📜 Code of Conduct
Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms. We are here to learn and build together.

---
**Note:** This project is periodically maintained. We appreciate your patience with PR reviews!