# Contactless Pulse Measurement with ROS

## Face Detection

You can use either the pylon-ros-camera or the video_stream_opencv package to provide image data for the face detection.

#### Install

```sh
git clone --recurse-submodules -j8 git@github.com:MobMonRob/PulsMeasurementStudien2.git
cd PulsMeasurementStudien2
catkin_make
```

If you want to use the industry camera, you have to install the following ros dependencies:

```sh
sudo sh -c 'echo "yaml https://raw.githubusercontent.com/basler/pylon-ros-camera/master/pylon_camera/rosdep/pylon_sdk.yaml" > /etc/ros/rosdep/sources.list.d/30-pylon_camera.list' && rosdep update && sudo rosdep install --from-paths . --ignore-src --rosdistro=$ROS_DISTRO -y
```

#### Run

```sh
source devel/setup.bash
# Run one of the following launch files
roslaunch ros_face_detection industry_camera.launch
roslaunch ros_face_detection webcam.launch
```
