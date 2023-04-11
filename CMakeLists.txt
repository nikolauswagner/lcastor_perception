cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(lcastor_perception)

find_package(catkin REQUIRED COMPONENTS
	roslaunch
	rospy
	message_generation

	geometry_msgs
	sensor_msgs
	std_msgs
	vision_msgs
)

catkin_python_setup()

add_message_files(FILES

)

add_service_files(FILES
  DetectObjects.srv
)

generate_messages(DEPENDENCIES
	sensor_msgs
	vision_msgs
)

catkin_package(
CATKIN_DEPENDS 
	roslaunch
	rospy
)

file(GLOB python_scripts_DIR "scripts/*.py")
file(GLOB bash_scripts_DIR "scripts/*.sh")
catkin_install_python(PROGRAMS
	src/lcastor_perception/object_detector.py
	src/lcastor_perception/object_manager.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
