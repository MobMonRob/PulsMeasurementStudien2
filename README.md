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

## Installation
The project is implemented in Python 2.7. and uses ROS Melodic. 

### Ubuntu
ROS recommends using Ubuntu 18.04, which you can find here http://releases.ubuntu.com/18.04.4/.

### ROS
For the installation of ROS Melodic, see http://wiki.ros.org/melodic/Installation/Ubuntu.

### Project
Clone the repository with it's submodules and run catkin_make in order to install project.
```sh
git clone --recurse-submodules -j8 https://github.com/MobMonRob/PulsMeasurementStudien2.git
cd PulsMeasurementStudien2
catkin_make
```
Note: In order to make catkin_make succeed, you need to have pylon installed on the computer (see next step).  
Alternatively you can delete the src/pylon-ros-camera directory, if you don't need the industry camera.

All python dependencies can be found in src/requirements.txt. Run the following in order to install them:
```sh
pip install -r src/requirements.txt
```

### Industry Camera Driver
Download and install the pylon driver https://www.baslerweb.com/de/vertrieb-support/downloads/downloads-software/.
After successfully installing pylon, you need to run the following in order to install some ROS dependencies:

```sh
sudo sh -c 'echo "yaml https://raw.githubusercontent.com/basler/pylon-ros-camera/master/pylon_camera/rosdep/pylon_sdk.yaml" > /etc/ros/rosdep/sources.list.d/30-pylon_camera.list' && rosdep update && sudo rosdep install --from-paths . --ignore-src --rosdistro=$ROS_DISTRO -y
```
For more information visit https://github.com/basler/pylon-ros-camera.

### PlotJuggler
If you want to display the output as graph, you need to install PlotJuggler.
```sh
sudo apt-get install ros-melodic-plotjuggler
```
If that does not work on your system, see https://github.com/facontidavide/PlotJuggler for other installation possibilities.  
If you don't want  to install PlotJuggler you can also just print the results to the console.

## Measure Pulse from Head movement

You can measure the pulse from your head movement by using the pulse_head_movement package.<br/>
The method is inspired by http://people.csail.mit.edu/balakg/pulsefromheadmotion.html

### Run
There are three ways to execute the measurement:
1. Only display the measured values of the contactless method of head movement
2. Compare the values from the contactless method with a ground truth from the polarH7 chest strap. 
The pulse value from the polarH7 is measured in the pulse_chest_strap package.
3. Compare the values from the contactless method with a ground truth from the MAHNOB HCI Tagging Database.

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
#### Compare pulse values with a video from the MAHNOB HCI Tagging Database
For comparison, the results are always displayed in Plotjuggler, so the installation of Plotjuggler is necessary here. This is currently only supported for the webcam.
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/ecg" topic_to_compare:="/pulse_head_movement" video_file:="<path_to_video_file>" bdf_file="<path_to_bdf_file>"
```

## Measure Pulse with Eulerian Motion Magnification (Changing Colour Intensity)

The pulse can be measured by amplifying and filtering subtle colour changes in the face. This method is inspired by the Eulerian Motion Magnification from http://people.csail.mit.edu/mrub/papers/vidmag.pdf.

### Run
This method can be run similarly to the head movement method:

#### Display pulse values only 
Only displaying the values from Eulerian Motion Magnification:
```sh
source devel/setup.bash
# Run one of the following launch files
# if you want to use the industry camera
roslaunch eulerian_motion_magnification industry_camera.launch
# if you want to use the webcam
roslaunch eulerian_motion_magnification webcam.launch
```
If you want to display the values in PlotJuggler, launch PlotJuggler and subscribe to topic ```eulerian_motion_magnification```

#### Compare pulse values with pulse values from polarH7
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/pulse_chest_strap" topic_to_compare:="/eulerian_motion_magnification"
```

#### Compare pulse values with a video from the MAHNOB HCI Tagging Database
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/ecg" topic_to_compare:="/eulerian_motion_magnification" video_file:="<path_to_video_file>" bdf_file="<path_to_bdf_file>"
```

#### Show processed image
If you want to display the processed image, set the launch argument to ```show_processed_image:=true```, it is false by default.

## Measure Pulse with the legacy method

This method was implemented by another team of students (https://github.com/MobMonRob/PulsMeasurementStudien) and integrated in ROS within this repository.

### Run
This method can be run similarly to the head movement method:

#### Display pulse values only 
Only displaying the values from legacy method:
```sh
source devel/setup.bash
# Run one of the following launch files
# if you want to use the industry camera
roslaunch legacy_measurement industry_camera.launch
# if you want to use the webcam
roslaunch legacy_measurement webcam.launch
```
If you want to display the values in PlotJuggler, launch PlotJuggler and subscribe to topic ```legacy_measurement```

#### Compare pulse values with pulse values from polarH7
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/pulse_chest_strap" topic_to_compare:="/legacy_measurement"
```

#### Compare pulse values with a video from the MAHNOB HCI Tagging Database
```sh
source devel/setup.bash
roslaunch common compare_pulse_values.launch topic:="/ecg" topic_to_compare:="/legacy_measurement" video_file:="<path_to_video_file>" bdf_file="<path_to_bdf_file>"
```
