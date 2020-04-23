# Contactless Pulse Measurement with ROS
This repository is a student research project from three students from the Baden-Wuerttemberg Corporate State University Karlsruhe.

The approach of the project is to provide methods for contactless pulse measurement.

To reach this approach, the repository contains different modules:
* The eulerian_motion_magnification package: This package contains a contactless pulse measurement method which measures the pulse from little color changes in the face. 
* The face_detection package: This is needed as the pulse is determined from the face of a person.
* The legacy_measurement package: This package contains the implementation of a previous work on this topic. 
* The pulse_chest_strap package: This package enables pulse measurement with the PolarH7 chest strap. It can be used as ground truth for the contactless method.
* The pulse_head_movement package: This package contains a contactless pulse measurement method which measures the pulse from little head movements.

The usage of the individual packages is described in the following sections.

The project is implemented in Python 2.7. and uses ROS Melodic. Supported OS is Ubuntu.<br/>
For the installation of ROS, see http://wiki.ros.org/melodic/Installation/Ubuntu

## Installation

```sh
git clone --recurse-submodules -j8 git@github.com:MobMonRob/PulsMeasurementStudien2.git
cd PulsMeasurementStudien2
catkin_make
```

If you want to use the industry camera, you have to install the following ros dependencies:

```sh
sudo sh -c 'echo "yaml https://raw.githubusercontent.com/basler/pylon-ros-camera/master/pylon_camera/rosdep/pylon_sdk.yaml" > /etc/ros/rosdep/sources.list.d/30-pylon_camera.list' && rosdep update && sudo rosdep install --from-paths . --ignore-src --rosdistro=$ROS_DISTRO -y
```

## Face Detection

You can use either the pylon-ros-camera, the video_stream_opencv package or a video file to provide image data for the face detection.

## Measure Pulse from Head movement

You can measure the pulse from your head movement by using the pulse_head_movement package.<br/>
The method is inspired by http://people.csail.mit.edu/balakg/pulsefromheadmotion.html

### Install
All python dependencies can be found in pulse_head_movement/requirements.txt

Furthermore if you want to display the output as graph, you need to install PlotJuggler.<br/>
For installation run 
```sh
sudo apt-get install ros-melodic-plotjuggler
```
If that does not work on your system, see https://github.com/facontidavide/PlotJuggler for other installation possibilities.<br/>
If you don't want  to install PlotJuggler you can also just print the results to the console.
### Run
There are two ways to execute the measurement:
1. Only display the measured values of the contactless method of head movement
2. Compare the values from the contacless method with a ground truth from the polarH7 chest strap. 
The pulse value from the polarH7 is measured in the pulse_chest_strap package.
#### Display pulse values only 
If you only want to get the pulse values from the contactless head movement method, you have two possibilities:
1. Print the pulse values to the console:
    ```sh
    source devel/setup.bash
    # Run one of the following launch files
    # if you want to use the industry camera
    roslaunch pulse_head_movement industry_camera.launch
    # if you want to use the webcam
    roslaunch pulse_head_movement webcam.launch
    ```
2. Show the pulse values in PlotJuggler:
    ```sh
    source devel/setup.bash
    # Run one of the following launch files
    # if you want to use the industry camera 
    roslaunch pulse_head_movement industry_camera.launch show_plot:=true
    # if you want to use the webcam
    roslaunch pulse_head_movement webcam.launch show_plot:=true
    ```
#### Compare pulse values with pulse values from polarH7
For comparison, the results are always displayed in Plotjuggler, so the installation of Plotjuggler is necessary here.
```sh
source devel/setup.bash
# Run one of the following launch files
# if you want to use the industry camera
roslaunch pulse_head_movement compare_industry_camera.launch
# if you want to use the webcam
roslaunch pulse_head_movement compare_webcam.launch
```


