[![CI](https://github.com/embedded-dev-research/itlab_2023/actions/workflows/ci.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2023/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/embedded-dev-research/itlab_2023/graph/badge.svg?token=L3OS8C4BI6)](https://codecov.io/gh/embedded-dev-research/itlab_2023)

# itlab_2023
# Name
## Short description ##
# **Building a Project**
### *Windows*
To build and run this project locally on Windows, follow these steps:

1. **Clone the repository:**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/embedded-dev-research/itlab_2023.git
   ```
2. **Update submodules:**
   Navigate to the project directory and update the submodules:
   ```bash
   git submodule update --init --recursive
3. **Configure the project:**
   Create a separate directory for configure the project and compile it:
   ```bash
   mkdir build
   cd build
   cmake ..
    ```
    *Note: Make sure you have CMake installed to build the project.*
4. **Build the project:**
   Next, to build the project, we will need to enter the command
    ```bash
   cmake --build . --config Release
    ```
    If you want to build in a debug, change the release to debug
5. **Run the project**
   After building the project, you can find the executable file in the following path from the *build* folder
   ```bash
   cd app\Release
    ```
   and run the file
    ```bash
   Reader.exe
    ```
### *Linux/macOS*
   To build and run this project locally on Linux or macOS, follow these steps:

1. **Clone the repository:**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/embedded-dev-research/itlab_2023.git
   ```
2. **Update submodules:**
   Navigate to the project directory and update the submodules:
   ```bash
   git submodule update --init --recursive
3. **Install necessary dependencies:**
   1. OpenMP
     Debian/Ubuntu:
     ```bash
     sudo apt-get install -y libomp-dev
     ```
     macOS:
     ```
     brew install libomp
     ```
4. **Configure the project:**
   Create a separate directory for configure the project and compile it:
   ```bash
   cmake -S . -B build
    ```
    *Note: Make sure you have CMake installed to build the project.*
   For macOS need to specify the path to omp.h file:
   ```bash
   cmake -S . -B build -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include" -DCMAKE_C_FLAGS="-I$(brew --prefix libomp)/include"
   ```
5. **Build the project:**
   Next, to build the project, we will need to enter the command
    ```bash
   cmake --build build --config Release
    ```
    If you want to build in a debug, change the release to debug
6. **Run the project**
   After building the project, you can find the executable file in the following path from the *build* folder
   ```bash
   cd build/app
    ```
   and run the file
    ```bash
   ./Reader
    ```
# Test Process
   This project contains tests to verify functionality.
   To test the project, the Google Test Framework is used as a submodule of the project.
   ### Google Test Framework

   Google Test is a powerful framework for unit testing in C++. In this project, Google Test is a submodule. When building the project, you have already       updated it, and it is ready for use.
   ### Running tests
   ### *Windows*
   
   To start the testing process locally, you need to go to the directory
   ```bash
   cd test\Release
   ```
   and run the following files:
   ```bash
   run_test.exe
   ```
### *Linux*
To start the testing process locally, you need to go to the directory
   ```bash
   cd build/test
   ```
   and run the following files:
   ```bash
   chmod +x run_test
   ./run_test
   ```
# **Some files used to create the library**
### *neural network models*
You need to download [Alexnet-model.h5](https://github.com/moizahmed97/Convolutional-Neural-Net-Designer/blob/master/AlexNet-model.h5) to the folder *docs*

# **How do I launch the inference?**
* You need to run the script *parcer.py* that is located in app/AlexNet to read weights from a model *Alexnet-model.h5* and the json file with the weights will be stored in the *docs* folder.
* Then put the test images in png format in the folder *docs/input*

# **Accuracy validation**
To run accuracy validation you need to use the MNIST dataset, which you can download [here](https://github.com/DeepTrackAI/MNIST_dataset/tree/main/mnist/test) and put it in a folder *docs/mnist/mnist/test*
Now you can run accuracy check - *build\bin\ACC_MNIST.exe*

# **Structure of our library**
![Class diagram](./docs/class_diagram.svg)
