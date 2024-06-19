FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Config
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV ROS_DISTRO noetic

# Install ROS
RUN rm /etc/apt/sources.list.d/* && \
    DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common lsb-release curl wget apt-transport-https git --no-install-recommends
RUN rm -rf /var/lib/apt/lists/* && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN DEBIAN_FRONTEND=noninteractive apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-catkin ros-${ROS_DISTRO}-ros-base ros-${ROS_DISTRO}-vision-msgs ros-${ROS_DISTRO}-cv-bridge python3-catkin-tools python3-venv python3-pip python3-rosdep --no-install-recommends && \
    rosdep init && rosdep update

# Install Tensorflow
RUN pip3 install tensorflow tensorflow-hub torchvision

# Install perception package
WORKDIR /home/lcastor/ros_ws/src
RUN git clone https://github.com/nikolauswagner/lcastor_perception.git
WORKDIR /home/lcastor/ros_ws/
RUN rosdep install --from-paths src --ignore-src -r -y
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin build

# Create entrypoint for image
COPY entrypoint.sh /
WORKDIR /home/lcastor/ros_ws
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]