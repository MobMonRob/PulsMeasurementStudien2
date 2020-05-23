# Contactless Pulse Measurement with ROS
This repository is a student research project from three students from the Baden-Wuerttemberg Corporate State University Karlsruhe.

The approach of the project is to provide methods for contactless pulse measurement.

To reach this approach, the repository contains different modules:
* The common package: This package contains common classes that are used in multiple packages like the face detector.
* The eulerian_motion_magnification package: This package contains a contactless pulse measurement method which measures the pulse from little color changes in the face. 
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
For comparison, the results are always displayed in Plotjuggler, so the installation of Plotjuggler is necessary here. This is currently only supported for the webcam.
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/pulse_chest_strap" topic_to_compare:="/pulse_head_movement"
```
#### Compare pulse values with a video from the MAHNOB-HCI-Tagging-Database
For comparison, the results are always displayed in Plotjuggler, so the installation of Plotjuggler is necessary here. This is currently only supported for the webcam.
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/ecg" topic_to_compare:="/pulse_head_movement" video_file:="<path_to_video_file>" bdf_file="<path_to_bdf_file>"
```
## Measure Pulse with Eulerian Motion Magnification (Changing Colour Intensity)
The pulse can be measured by amplifying and filtering subtle colour changes in the face. This method is inspired by the Eulerian Motion Magnification from http://people.csail.mit.edu/mrub/papers/vidmag.pdf.

### Install
Install PlotJuggler as mentioned above.

### Run
This method can be run similarly to the head movement method:

#### Display pulse values only 
Only displaying the values from Eulerian Motion Magnification:
Print the pulse values to the console:
    ```sh
    source devel/setup.bash
    # Run one of the following launch files
    # if you want to use the industry camera
    roslaunch eulerian_motion_magnification industry_camera.launch
    # if you want to use the webcam
    roslaunch eulerian_motion_magnification webcam.launch
    ```
If you want to display the values in PlotJuggler, launch PlotJuggler and subscribe to topic eulerian_motion_magnification/pulse

#### Compare pulse values with pulse values from polarH7
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/pulse_chest_strap" topic_to_compare:="/eulerian_motion_magnification"
```

#### Compare pulse values with a video from the MAHNOB-HCI-Tagging-Database
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/ecg" topic_to_compare:="/eulerian_motion_magnification" video_file:="<path_to_video_file>" bdf_file="<path_to_bdf_file>"
```

#### Show processed image
If you want to display the processed image, set property in launch/eulerian_motion_magnification.launch to:
```sh
<arg name="show_processed_image" default="true" />
```
otherwise set false.
